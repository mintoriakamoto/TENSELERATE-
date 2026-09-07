"""
The decode loop.

Spec rule for this module: no blocking calls in the hot path. The loop below is
pure compute over in-memory state — no I/O, no locks, no synchronous waits — so
that when the CUDA backend lands, each step is a launch sequence and nothing on
the host stalls the stream. Sampling is greedy or temperature; the structure
(prefill once, then one token per step carrying KV/GDN state forward) is the
contract the batched scheduler will parallelize across sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from tenselerate import nvtx
from tenselerate.reference.model import LayerState, ReferenceModel


@dataclass
class SamplingParams:
    """Parameters controlling text generation behavior.

    Attributes:
        max_tokens: Maximum number of tokens to generate (default 64).
        temperature: Sampling temperature. 0.0 = greedy (argmax). > 0.0 = sample
            from logits divided by temperature (the model card default is 0.0).
        repeat_penalty: Penalty for repeating tokens. Logits of seen tokens are
            divided by this value; unseen tokens are multiplied by it (default 1.15).
        seed: RNG seed for deterministic sampling (default 0).
        stop_tokens: Token ids that stop generation early when encountered.
    """
    max_tokens: int = 64
    temperature: float = 0.0
    repeat_penalty: float = 1.15
    seed: int = 0
    stop_tokens: tuple[int, ...] = ()


def _sample(logits: np.ndarray, params: SamplingParams,
            seen: dict[int, int], rng: np.random.Generator) -> int:
    """Sample the next token from logits with repeat penalty and temperature.

    Applies repeat penalty to logits of tokens already seen in the sequence,
    then samples greedily (argmax) or from the softmax distribution depending
    on temperature.

    Args:
        logits: Unnormalized log probabilities for each token (vocab_size,).
        params: Sampling parameters (temperature, repeat_penalty, etc).
        seen: Dict mapping token id to count of times it has appeared.
        rng: NumPy random generator for sampling.

    Returns:
        The sampled token id (0 <= id < vocab_size).
    """
    logits = logits.astype(np.float64).copy()
    if params.repeat_penalty != 1.0 and seen:
        idx = np.fromiter(seen.keys(), dtype=np.int64)
        pos = logits[idx] > 0
        logits[idx[pos]] /= params.repeat_penalty
        logits[idx[~pos]] *= params.repeat_penalty
    if params.temperature <= 0.0:
        return int(np.argmax(logits))
    logits /= params.temperature
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


class Generator:
    """Single-sequence generation over the reference model (backend bring-up).

    The generator runs the model one token at a time, maintaining layer state
    (KV cache, GDN recurrent state) across steps. This is the oracle for validating
    kernels and backends. No batching yet — the scheduler will parallelize this
    across sequences when it lands.
    """

    def __init__(self, model: ReferenceModel):
        """Initialize the generator with a reference model.

        Args:
            model: ReferenceModel instance to run inference on.
        """
        self.model = model

    def prefill(self, prompt: list[int], state: list[LayerState]) -> tuple[np.ndarray, int]:
        """Process the prompt and return logits for the next position.

        Runs the entire prompt through the model in a single prefill phase,
        updating state in place. This is efficient for long prompts before
        switching to one-token-per-step decode.

        Args:
            prompt: List of token ids (must be non-empty).
            state: Layer state list (KV cache, GDN state) to update in place.

        Returns:
            Tuple of (logits_for_next_position, position_after_prefill).

        Raises:
            ValueError: If prompt is empty.
        """
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        logits = self.model.step(prompt[0], 0, state)
        with nvtx.range("prefill"):
            for pos in range(1, len(prompt)):
                logits = self.model.step(prompt[pos], pos, state)
        return logits, len(prompt)

    def generate(self, prompt: list[int], params: SamplingParams) -> Iterator[int]:
        """Generate tokens one at a time with the given sampling parameters.

        Creates fresh state, prefills the prompt, then yields tokens one at a time
        until max_tokens is reached or a stop token is sampled. State (KV, GDN)
        is carried forward across steps.

        Args:
            prompt: Initial prompt tokens (must be non-empty).
            params: Sampling parameters (max_tokens, temperature, etc).

        Yields:
            Generated token ids one at a time.

        Raises:
            ValueError: If prompt is empty.
        """
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        state = self.model.new_state()
        rng = np.random.default_rng(params.seed)
        seen: dict[int, int] = {t: 1 for t in prompt}

        logits, pos = self.prefill(prompt, state)
        for _ in range(params.max_tokens):
            tok = _sample(logits, params, seen, rng)
            if tok in params.stop_tokens:
                return
            yield tok
            seen[tok] = seen.get(tok, 0) + 1
            with nvtx.range("decode_step"):
                logits = self.model.step(tok, pos, state)
            pos += 1

    def generate_list(self, prompt: list[int], params: SamplingParams) -> list[int]:
        """Generate tokens and return as a list (blocking variant of generate).

        Convenience wrapper around generate() for cases where the full output
        is needed at once.

        Args:
            prompt: Initial prompt tokens (must be non-empty).
            params: Sampling parameters (max_tokens, temperature, etc).

        Returns:
            List of generated token ids.
        """
        return list(self.generate(prompt, params))


def make_streamer(on_token: Callable[[int], None]):
    """Create a callback adapter for token-by-token streaming.

    Useful for servers that need to process tokens as they are generated
    (e.g., send them to a client) while still collecting the full output.

    Args:
        on_token: Callback function called with each generated token id.

    Returns:
        A function that takes a token iterator, calls on_token for each token,
        and returns the full list of tokens.
    """
    def run(gen: Iterator[int]) -> list[int]:
        out = []
        for t in gen:
            out.append(t)
            on_token(t)
        return out
    return run
