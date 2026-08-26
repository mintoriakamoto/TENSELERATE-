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
    max_tokens: int = 64
    temperature: float = 0.0          # 0 = greedy (the model card's default)
    repeat_penalty: float = 1.15
    seed: int = 0
    stop_tokens: tuple[int, ...] = ()


def _sample(logits: np.ndarray, params: SamplingParams,
            seen: dict[int, int], rng: np.random.Generator) -> int:
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
    """Single-sequence generation over the reference model (backend bring-up)."""

    def __init__(self, model: ReferenceModel):
        self.model = model

    def prefill(self, prompt: list[int], state: list[LayerState]) -> tuple[np.ndarray, int]:
        """Run the prompt through, returning last-token logits and next position."""
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        logits = self.model.step(prompt[0], 0, state)
        with nvtx.range("prefill"):
            for pos in range(1, len(prompt)):
                logits = self.model.step(prompt[pos], pos, state)
        return logits, len(prompt)

    def generate(self, prompt: list[int], params: SamplingParams) -> Iterator[int]:
        """Yield generated token ids one at a time. Prompt must be non-empty."""
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
        return list(self.generate(prompt, params))


def make_streamer(on_token: Callable[[int], None]):
    """Adapter for a server that wants a callback per token."""
    def run(gen: Iterator[int]) -> list[int]:
        out = []
        for t in gen:
            out.append(t)
            on_token(t)
        return out
    return run
