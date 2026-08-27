"""
The kernel bridge: calls the compiled CUDA int8 GEMM through ctypes, with the
NumPy reference (tenselerate/reference/numerics.py) as the automatic fallback
when no CUDA build is present.

This is deliberately a thin ctypes layer, not a C extension module, so nothing
in the engine needs a compile step to import - only to go fast. The shared
library (`libtenselerate_int8_gemm_c.so`, built by tenselerate/csrc/CMakeLists.txt
only when a CUDA compiler is present) is discovered by searching this repo's
`build-*` directories, or pinned exactly via `TENSELERATE_INT8_GEMM_LIB` - which
is also how a test points this module at a CPU-only stub .so with the identical
C ABI, to validate the ENTIRE marshaling path (pointer types, contiguity,
error-code handling) without a GPU. See tests/tenselerate/test_backend_bridge.py.

Every backend implements the same interface as
`reference.numerics.quantized_linear`, so model code can call through this
module unconditionally and get the fastest thing available on the machine it
runs on.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from tenselerate.reference import numerics as nx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LIB_NAME = "libtenselerate_int8_gemm_c"
ENV_OVERRIDE = "TENSELERATE_INT8_GEMM_LIB"

# candidate build directories, in the order CI and local builds are named
_BUILD_DIRS = ("build-cuda", "build-kernels", "build")


class GemmBackend(Protocol):
    """The interface every int8 GEMM implementation provides."""
    name: str

    def matmul(self, a_q: NDArray[np.int8], a_scale: NDArray[np.float32],
               b_q: NDArray[np.int8], b_scale: NDArray[np.float32]) -> NDArray[np.float32]:
        ...


class ReferenceBackend:
    """Pure NumPy. Always available; the correctness oracle for the CUDA path."""
    name = "reference"

    def matmul(self, a_q, a_scale, b_q, b_scale):
        return nx.int8_matmul(a_q, a_scale.reshape(-1, 1), b_q, b_scale.reshape(-1, 1))


class CudaGemmError(RuntimeError):
    """A loaded CUDA GEMM call returned a non-zero status."""


class CudaBackend:
    """
    ctypes binding to `tenselerate_int8_gemm` (see csrc/int8_gemm.h). Host
    pointers only - the native side owns the device malloc/copy/launch/free, so
    this class never touches CUDA APIs directly and needs no GPU-specific
    Python dependency.
    """
    name = "cuda"

    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.tenselerate_int8_gemm.argtypes = [
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self._lib.tenselerate_int8_gemm.restype = ctypes.c_int

    def matmul(self, a_q, a_scale, b_q, b_scale):
        a_q = np.ascontiguousarray(a_q, dtype=np.int8)
        b_q = np.ascontiguousarray(b_q, dtype=np.int8)
        a_scale = np.ascontiguousarray(a_scale, dtype=np.float32).reshape(-1)
        b_scale = np.ascontiguousarray(b_scale, dtype=np.float32).reshape(-1)
        M, K = a_q.shape
        N, K2 = b_q.shape
        if K != K2:
            raise ValueError(f"K mismatch: A has {K}, B has {K2}")
        if a_scale.shape != (M,):
            raise ValueError(f"a_scale must be shape ({M},), got {a_scale.shape}")
        if b_scale.shape != (N,):
            raise ValueError(f"b_scale must be shape ({N},), got {b_scale.shape}")

        c = np.empty((M, N), dtype=np.float32)
        rc = self._lib.tenselerate_int8_gemm(
            a_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            a_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            b_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        )
        if rc != 0:
            raise CudaGemmError(
                f"tenselerate_int8_gemm returned {rc} "
                f"({self.lib_path.name}, M={M} N={N} K={K})")
        return c


def find_shared_lib() -> Path | None:
    """
    Locate the compiled CUDA shared library. `TENSELERATE_INT8_GEMM_LIB` wins
    outright (an explicit path, used by tests and deliberate deployments);
    otherwise search this repo's known build directories for the platform's
    shared-library naming (.so / .dylib / .dll).
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        p = Path(override)
        return p if p.is_file() else None

    for ext in (".so", ".dylib", ".dll"):
        for d in _BUILD_DIRS:
            candidate = REPO_ROOT / d / f"{LIB_NAME}{ext}"
            if candidate.is_file():
                return candidate
    return None


def available() -> bool:
    """True if a CUDA GEMM shared library was found (does not guarantee a GPU
    at runtime - a load failure inside get_backend() still falls back)."""
    return find_shared_lib() is not None


def get_backend(force: str | None = None) -> GemmBackend:
    """
    The fastest backend available, unless `force` names one explicitly
    ("cuda" or "reference") - used by tests and by callers that must not
    silently degrade to the slow path.
    """
    if force == "reference":
        return ReferenceBackend()
    if force == "cuda":
        lib = find_shared_lib()
        if lib is None:
            raise FileNotFoundError(
                f"no CUDA GEMM library found (checked {ENV_OVERRIDE} and "
                f"{_BUILD_DIRS}); build tenselerate/csrc with CUDA available")
        return CudaBackend(lib)
    if force is not None:
        raise ValueError(f"unknown backend {force!r}, expected 'cuda' or 'reference'")

    lib = find_shared_lib()
    if lib is None:
        return ReferenceBackend()
    try:
        return CudaBackend(lib)
    except OSError:
        # the .so exists but failed to load (wrong platform, missing CUDA
        # runtime libs at load time, ...) - degrade rather than crash import
        return ReferenceBackend()


def quantized_linear(x: NDArray[np.float32], w: NDArray[np.float32],
                     backend: GemmBackend | None = None) -> NDArray[np.float32]:
    """
    Drop-in replacement for reference.numerics.quantized_linear that runs the
    matmul on `backend` (the fastest available, if not given). Quantization
    stays on the host either way - only the GEMM itself moves to the backend -
    so this is the same op the model calls, just able to go through CUDA.
    """
    if backend is None:
        backend = get_backend()
    xq, xs = nx.quantize_int8_symmetric(x, axis=-1)
    wq, ws = nx.quantize_int8_symmetric(w, axis=-1)
    return backend.matmul(xq, xs.reshape(-1), wq, ws.reshape(-1))
