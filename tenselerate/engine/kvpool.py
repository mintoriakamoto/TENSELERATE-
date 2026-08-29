"""
Paged KV block pool for the windowed full-attention layers.

The idea is vLLM's PagedAttention, in our own code and adapted to this engine's
two constraints:

  * only the 16 full-attention layers cache at all - the 48 Gated-DeltaNet layers
    keep a fixed per-sequence state, allocated separately and never paged;
  * the full-attention layers are WINDOWED, so a sequence's KV stops growing once
    it reaches the window. Blocks past the window are recycled, which is what
    lets a 1,000,000-token sequence hold a bounded, constant amount of KV.

Blocks are fixed-size and non-contiguous, so a sequence never needs a contiguous
run of memory and the pool does not fragment. Capacity is derived from a real
VRAM budget, so admission control can refuse work the GPU cannot hold instead of
discovering it at allocation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GiB = 1024 ** 3
DEFAULT_BLOCK_TOKENS = 256


class OutOfBlocks(RuntimeError):
    """The pool cannot satisfy an allocation. Callers must not swallow this."""


@dataclass
class KVBlockPool:
    """
    A fixed set of equally-sized KV blocks.

    block_tokens : tokens of KV held by one block
    n_blocks     : total blocks in the pool
    """
    block_tokens: int
    n_blocks: int
    _free: list[int] = field(default_factory=list, init=False, repr=False)
    _allocated: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.block_tokens <= 0 or self.n_blocks <= 0:
            raise ValueError("block_tokens and n_blocks must be positive")
        # hand out low ids first; deterministic makes tests meaningful
        self._free = list(range(self.n_blocks - 1, -1, -1))

    # -- capacity ----------------------------------------------------------
    @classmethod
    def from_budget(cls, budget_bytes: float, kv_bytes_per_token: float,
                    block_tokens: int = DEFAULT_BLOCK_TOKENS) -> "KVBlockPool":
        """Size the pool from a VRAM budget rather than a guessed block count."""
        per_block = kv_bytes_per_token * block_tokens
        n = int(budget_bytes // per_block)
        if n <= 0:
            raise ValueError(
                f"budget {budget_bytes / GiB:.2f} GiB holds no blocks of "
                f"{block_tokens} tokens ({per_block / GiB:.3f} GiB each)")
        return cls(block_tokens=block_tokens, n_blocks=n)

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_allocated(self) -> int:
        return self._allocated

    def blocks_for_tokens(self, tokens: int) -> int:
        """Blocks needed to hold `tokens` (ceiling division)."""
        return (tokens + self.block_tokens - 1) // self.block_tokens

    # -- allocation --------------------------------------------------------
    def allocate(self, n: int = 1) -> list[int]:
        if n <= 0:
            return []
        if n > self.n_free:
            raise OutOfBlocks(f"need {n} blocks, {self.n_free} free")
        out = [self._free.pop() for _ in range(n)]
        self._allocated += n
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if not 0 <= b < self.n_blocks:
                raise ValueError(f"block {b} is not from this pool")
            self._free.append(b)
        self._allocated -= len(blocks)

    def can_allocate(self, n: int) -> bool:
        return n <= self.n_free
