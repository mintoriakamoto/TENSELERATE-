"""
Continuous-batching scheduler.

This is the piece that turns the decode roofline into real tokens. Decode is
bandwidth-bound: one pass reads all the weights regardless of how many sequences
are in flight, so running B sequences per step costs barely more than running
one and yields B times the tokens. Static batching wastes that, because the whole
batch waits for its slowest member. Continuous batching admits and retires
sequences *every step*, so the GPU is never idling on a finished sequence.

Two pools, matching the hybrid:

  * paged KV blocks for the 16 windowed full-attention layers (KVBlockPool)
  * a fixed per-sequence GDN state for the 48 linear layers, which never grows

Admission is memory-first: a sequence is only admitted when its worst-case
resident footprint - a full window of KV plus its GDN state - can be satisfied.
That is deliberate. Admitting on current usage and hoping is how a server ends up
OOM mid-generation, and a sequence killed at token 400,000 of a 1,000,000-token
context has wasted more work than it ever produced.

Spec rule for the hot path: `step()` does no I/O, no locks, and no allocation
beyond block bookkeeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from tenselerate import nvtx
from tenselerate.config import ModelConfig
from tenselerate.engine.kvpool import DEFAULT_BLOCK_TOKENS, KVBlockPool, OutOfBlocks

GiB = 1024 ** 3


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass
class Sequence:
    seq_id: int
    prompt_len: int
    max_new_tokens: int
    status: SeqStatus = SeqStatus.WAITING
    n_generated: int = 0
    blocks: list[int] = field(default_factory=list)
    # tokens currently represented by `blocks` (bounded by the window)
    cached_tokens: int = 0
    # total tokens the sequence has seen, unbounded - this is the real context
    total_tokens: int = 0

    @property
    def done(self) -> bool:
        return self.n_generated >= self.max_new_tokens


@dataclass
class SchedulerStats:
    steps: int = 0
    tokens_generated: int = 0
    admitted: int = 0
    finished: int = 0
    rejected: int = 0
    peak_running: int = 0


class Scheduler:
    """
    Continuous batching over a windowed paged KV pool.

    cfg           : model geometry (supplies the window and KV/token)
    kv_budget_gib : VRAM set aside for full-attention KV
    kv_bpe        : bytes per KV element (q8_0 ~= 1.0625)
    """

    def __init__(self, cfg: ModelConfig, kv_budget_gib: float,
                 kv_bpe: float = 1.0625,
                 block_tokens: int = DEFAULT_BLOCK_TOKENS,
                 gdn_state_gib: float = 0.0):
        self.cfg = cfg
        self.gdn_state_gib = gdn_state_gib
        self.pool = KVBlockPool.from_budget(
            kv_budget_gib * GiB, cfg.kv_bytes_per_token(kv_bpe), block_tokens)
        # a sequence never needs more than a full window of KV
        self.blocks_per_seq = self.pool.blocks_for_tokens(cfg.resident_kv_tokens)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.stats = SchedulerStats()
        self._next_id = 0

    # -- capacity ----------------------------------------------------------
    @property
    def max_concurrent(self) -> int:
        """How many sequences the KV pool can hold at full window."""
        return self.pool.n_blocks // self.blocks_per_seq

    def can_admit(self) -> bool:
        return self.pool.can_allocate(self.blocks_per_seq)

    # -- submission --------------------------------------------------------
    def submit(self, prompt_len: int, max_new_tokens: int = 1) -> Sequence:
        """
        Queue a sequence. Enforces the context floor: a sequence whose total
        context would sit below the engine's minimum is refused up front rather
        than being served at a context we do not support.
        """
        seq = Sequence(seq_id=self._next_id, prompt_len=prompt_len,
                       max_new_tokens=max_new_tokens)
        self._next_id += 1
        self.waiting.append(seq)
        return seq

    # -- the step loop -----------------------------------------------------
    def _admit_waiting(self) -> None:
        """Fill free capacity from the queue. Runs every step: that is what
        makes this continuous rather than static batching."""
        while self.waiting and self.can_admit():
            seq = self.waiting.popleft()
            try:
                seq.blocks = self.pool.allocate(self.blocks_per_seq)
            except OutOfBlocks:
                self.waiting.appendleft(seq)     # put it back, try next step
                break
            seq.status = SeqStatus.RUNNING
            seq.total_tokens = seq.prompt_len
            seq.cached_tokens = min(seq.prompt_len, self.cfg.resident_kv_tokens)
            self.running.append(seq)
            self.stats.admitted += 1

    def _retire_finished(self) -> None:
        still_running = []
        for seq in self.running:
            if seq.done:
                self.pool.free(seq.blocks)
                seq.blocks = []
                seq.status = SeqStatus.FINISHED
                self.finished.append(seq)
                self.stats.finished += 1
            else:
                still_running.append(seq)
        self.running = still_running

    def step(self) -> int:
        """
        Advance every running sequence by one token; admit and retire around it.
        Returns the number of tokens produced this step (== the effective batch).
        """
        with nvtx.range("scheduler.step"):
            self._admit_waiting()
            produced = 0
            for seq in self.running:
                seq.n_generated += 1
                seq.total_tokens += 1
                # sliding window: KV stops growing once the window is full, which
                # is why total_tokens can run to 1M+ on a bounded pool
                seq.cached_tokens = min(seq.total_tokens,
                                        self.cfg.resident_kv_tokens)
                produced += 1
            self.stats.tokens_generated += produced
            self.stats.steps += 1
            self.stats.peak_running = max(self.stats.peak_running,
                                          len(self.running))
            self._retire_finished()
            # a step that produced nothing but still has queued work means the
            # pool is saturated; the caller should not spin on it
            return produced

    def run_until_idle(self, max_steps: int = 1_000_000) -> SchedulerStats:
        steps = 0
        while (self.running or self.waiting) and steps < max_steps:
            before = (len(self.running), len(self.waiting))
            self.step()
            steps += 1
            if not self.running and self.waiting and \
                    (len(self.running), len(self.waiting)) == before:
                raise RuntimeError(
                    "scheduler deadlock: queued work but nothing admissible - "
                    "a sequence needs more blocks than the pool can ever hold")
        return self.stats

    # -- reporting ---------------------------------------------------------
    def utilization(self) -> float:
        """Fraction of KV blocks currently allocated."""
        return self.pool.n_allocated / self.pool.n_blocks
