"""
Minimal GGUF writer.

Enough to emit a valid GGUF the reader round-trips against in tests, and a
starting point for exporting TENSELERATE-native checkpoints later. It writes the
metadata types the engine cares about plus F32/F16/Q8_0 tensors. It is not a
general quantizer — quantized export grows with the engine.
"""

from __future__ import annotations

import struct
from typing import Any, BinaryIO

import numpy as np

from tenselerate.gguf.reader import (
    GGUF_ARRAY, GGUF_BOOL, GGUF_DEFAULT_ALIGNMENT, GGUF_F32, GGUF_I32,
    GGUF_MAGIC, GGUF_STRING, GGUF_U32, GGUF_U64,
)


def _w_string(f: BinaryIO, s: str) -> None:
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def _w_value(f: BinaryIO, vtype: int, value: Any) -> None:
    if vtype == GGUF_STRING:
        _w_string(f, value)
    elif vtype == GGUF_U32:
        f.write(struct.pack("<I", value))
    elif vtype == GGUF_I32:
        f.write(struct.pack("<i", value))
    elif vtype == GGUF_F32:
        f.write(struct.pack("<f", value))
    elif vtype == GGUF_BOOL:
        f.write(struct.pack("<?", value))
    elif vtype == GGUF_U64:
        f.write(struct.pack("<Q", value))
    elif vtype == GGUF_ARRAY:
        elem_type, items = value
        f.write(struct.pack("<I", elem_type))
        f.write(struct.pack("<Q", len(items)))
        for it in items:
            _w_value(f, elem_type, it)
    else:
        raise ValueError(f"writer does not support value type {vtype}")


_NP_TO_GGML = {"F32": 0, "F16": 1, "Q8_0": 8}


def _encode_tensor(arr: np.ndarray, type_name: str) -> tuple[int, bytes]:
    if type_name == "F32":
        return 0, arr.astype("<f4").tobytes()
    if type_name == "F16":
        return 1, arr.astype("<f2").tobytes()
    if type_name == "Q8_0":
        flat = arr.astype(np.float32).reshape(-1)
        assert flat.size % 32 == 0, "Q8_0 needs a multiple of 32 elements"
        blocks = flat.reshape(-1, 32)
        amax = np.max(np.abs(blocks), axis=1, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(blocks / scale), -127, 127).astype(np.int8)
        out = bytearray()
        s16 = scale.reshape(-1).astype("<f2")
        for i in range(blocks.shape[0]):
            out += s16[i].tobytes()
            out += q[i].tobytes()
        return 8, bytes(out)
    raise ValueError(f"writer cannot encode {type_name}")


def write_gguf(
    path: str,
    metadata: dict[str, tuple[int, Any]],
    tensors: dict[str, tuple[np.ndarray, str]],
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
) -> None:
    """
    metadata: {key: (value_type, value)}
    tensors:  {name: (array, type_name)}   type_name in F32/F16/Q8_0
    Arrays are written in the given (row-major) order; GGUF dims are fastest-first
    so we store shape reversed, matching what the reader hands back.
    """
    encoded = {name: _encode_tensor(a, tn) for name, (a, tn) in tensors.items()}

    with open(path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", 3))                       # version
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))
        for key, (vtype, value) in metadata.items():
            _w_string(f, key)
            f.write(struct.pack("<I", vtype))
            _w_value(f, vtype, value)

        offset = 0
        blobs = []
        for name, (arr, tn) in tensors.items():
            ggml_type, blob = encoded[name]
            _w_string(f, name)
            dims = tuple(reversed(arr.shape))               # fastest-first
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", offset))
            blobs.append(blob)
            offset += len(blob)

        pos = f.tell()
        pad = (pos + alignment - 1) // alignment * alignment - pos
        f.write(b"\x00" * pad)
        for blob in blobs:
            f.write(blob)
