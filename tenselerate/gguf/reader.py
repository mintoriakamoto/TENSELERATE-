"""
GGUF reader — the TENSELERATE engine's own loader for real model files.

GGUF is the de-facto container for local models (it is what the RavenX Chaos
Agent ships as). We read it with our own code rather than depending on
llama.cpp's `gguf-py`, so the engine stands alone: nothing at runtime is
borrowed from another project, only the on-disk format is shared.

This parses the full header — magic, version, every metadata key/value, and the
tensor directory (name, shape, ggml type, offset) — and exposes each tensor's
raw bytes plus a float32 dequantization for the un-blocked types (F32, F16,
Q8_0). The k-quant block formats (Q4_K, Q6_K) that make up the RavenX body are
declared here with their block sizes and left as an explicit next step; the
reader already tells you exactly which types a file uses so nothing is silent.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np

GGUF_MAGIC = 0x46554747          # "GGUF" little-endian
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value type tags
(GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32,
 GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64) = range(13)

_SCALAR = {
    GGUF_U8: ("<B", 1), GGUF_I8: ("<b", 1), GGUF_U16: ("<H", 2),
    GGUF_I16: ("<h", 2), GGUF_U32: ("<I", 4), GGUF_I32: ("<i", 4),
    GGUF_F32: ("<f", 4), GGUF_BOOL: ("<?", 1), GGUF_U64: ("<Q", 8),
    GGUF_I64: ("<q", 8), GGUF_F64: ("<d", 8),
}

# ggml tensor type -> (block size in elements, bytes per block). Only the types
# we can already dequantize have a decoder below; the rest are recorded so a
# caller sees precisely what a model needs.
GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16",
}
GGML_BLOCK = {   # (elements per block, bytes per block)
    0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 14: (256, 210), 30: (1, 2),
}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]        # GGUF stores dims fastest-first; kept as read
    ggml_type: int
    offset: int                   # bytes from the start of the tensor-data region

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAME.get(self.ggml_type, f"TYPE_{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def nbytes(self) -> int:
        block_elems, block_bytes = GGML_BLOCK.get(self.ggml_type, (0, 0))
        if block_elems == 0:
            raise ValueError(f"unknown block size for {self.type_name}")
        return self.n_elements // block_elems * block_bytes


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of GGUF file")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALAR:
        fmt, _ = _SCALAR[vtype]
        return _read(f, fmt)
    if vtype == GGUF_STRING:
        return _read_string(f)
    if vtype == GGUF_ARRAY:
        elem_type = _read(f, "<I")
        count = _read(f, "<Q")
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {vtype}")


class GGUFReader:
    """Parsed GGUF header. Open a path, inspect metadata and tensors."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self._data_start = 0
        with open(path, "rb") as f:
            self._parse(f)

    def _parse(self, f: BinaryIO) -> None:
        magic = _read(f, "<I")
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic {magic:#x})")
        self.version = _read(f, "<I")
        if self.version != 3:
            raise ValueError(f"unsupported GGUF version {self.version}")
        n_tensors = _read(f, "<Q")
        n_kv = _read(f, "<Q")

        for _ in range(n_kv):
            key = _read_string(f)
            vtype = _read(f, "<I")
            self.metadata[key] = _read_value(f, vtype)
        self.alignment = int(self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))

        infos = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = _read(f, "<I")
            shape = tuple(_read(f, "<Q") for _ in range(n_dims))
            ggml_type = _read(f, "<I")
            offset = _read(f, "<Q")
            infos.append(TensorInfo(name, shape, ggml_type, offset))

        # tensor data starts after the header, padded up to `alignment`
        pos = f.tell()
        self._data_start = (pos + self.alignment - 1) // self.alignment * self.alignment
        self.tensors = {t.name: t for t in infos}

    # -- accessors -----------------------------------------------------------
    def type_histogram(self) -> dict[str, int]:
        """How many tensors of each ggml type — tells you what a model needs."""
        hist: dict[str, int] = {}
        for t in self.tensors.values():
            hist[t.type_name] = hist.get(t.type_name, 0) + 1
        return hist

    def raw_tensor(self, name: str) -> bytes:
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self._data_start + t.offset)
            return f.read(t.nbytes())

    def dequantize(self, name: str) -> np.ndarray:
        """Return a tensor as float32 in GGUF (fastest-first) order, flat."""
        t = self.tensors[name]
        raw = self.raw_tensor(name)
        if t.ggml_type == 0:                                  # F32
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if t.ggml_type == 1:                                  # F16
            return np.frombuffer(raw, dtype="<f2").astype(np.float32)
        if t.ggml_type == 30:                                 # BF16
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            return (u16 << 16).view(np.float32).astype(np.float32)
        if t.ggml_type == 8:                                  # Q8_0
            return _dequant_q8_0(raw, t.n_elements)
        raise NotImplementedError(
            f"dequant for {t.type_name} not implemented yet "
            f"(tensor {name!r}); type histogram: {self.type_histogram()}")


def _dequant_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    """
    Q8_0: blocks of 32 int8 with one f16 scale. Block layout is
    [f16 scale][32 x int8], 34 bytes, matching GGML_BLOCK[8].
    """
    n_blocks = n_elements // 32
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 34)
    scales = buf[:, :2].copy().view("<f2").astype(np.float32).reshape(n_blocks, 1)
    qs = buf[:, 2:].view(np.int8).astype(np.float32)
    return (qs * scales).reshape(-1)
