"""
The kernel bridge: ctypes marshaling, backend dispatch, and graceful fallback
when no CUDA build is present.

No GPU is required. `int8_gemm_stub.cpp` implements the CUDA shared library's
exact C ABI on the CPU (compiled here with plain g++), so pointing the ctypes
bridge at it exercises the real marshaling path end to end - the same code that
will load the actual CUDA .so on the target machines.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from tenselerate.backend import int8_gemm as bridge
from tenselerate.reference import numerics as nx

f32 = np.float32
CSRC = bridge.REPO_ROOT / "tenselerate" / "csrc"

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not available to build the test stub")


@pytest.fixture(scope="module")
def stub_lib(tmp_path_factory):
    """Compile int8_gemm_stub.cpp into a shared library, once per test module."""
    out_dir = tmp_path_factory.mktemp("bridge_stub")
    ext = "dylib" if sys.platform == "darwin" else "so"
    out = out_dir / f"{bridge.LIB_NAME}.{ext}"
    cmd = ["g++", "-shared", "-fPIC", "-O2", "-std=c++17",
           "-I", str(CSRC), str(CSRC / "int8_gemm_stub.cpp"), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"stub build failed:\n{proc.stderr}"
    assert out.is_file()
    return out


# ---- backend discovery ----------------------------------------------------
def test_no_lib_found_in_a_clean_environment(monkeypatch):
    monkeypatch.delenv(bridge.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(bridge, "REPO_ROOT", bridge.REPO_ROOT / "nonexistent")
    assert bridge.find_shared_lib() is None
    assert bridge.available() is False


def test_env_override_wins_and_is_validated(monkeypatch, stub_lib):
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    assert bridge.find_shared_lib() == stub_lib
    assert bridge.available() is True


def test_env_override_pointing_nowhere_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv(bridge.ENV_OVERRIDE, "/does/not/exist.so")
    assert bridge.find_shared_lib() is None


# ---- backend dispatch -------------------------------------------------
def test_get_backend_defaults_to_reference_with_nothing_available(monkeypatch):
    monkeypatch.delenv(bridge.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(bridge, "REPO_ROOT", bridge.REPO_ROOT / "nonexistent")
    be = bridge.get_backend()
    assert be.name == "reference"


def test_get_backend_picks_up_the_stub_as_if_it_were_cuda(monkeypatch, stub_lib):
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    be = bridge.get_backend()
    assert be.name == "cuda"


def test_force_cuda_without_a_lib_raises_rather_than_degrading(monkeypatch):
    monkeypatch.delenv(bridge.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(bridge, "REPO_ROOT", bridge.REPO_ROOT / "nonexistent")
    with pytest.raises(FileNotFoundError):
        bridge.get_backend(force="cuda")


def test_force_reference_ignores_an_available_lib(monkeypatch, stub_lib):
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    be = bridge.get_backend(force="reference")
    assert be.name == "reference"


def test_unknown_force_value_rejected():
    with pytest.raises(ValueError):
        bridge.get_backend(force="tpu")


# ---- the reference backend ------------------------------------------------
def test_reference_backend_matches_numerics_directly():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 64)).astype(f32)
    w = rng.standard_normal((6, 64)).astype(f32)
    aq, as_ = nx.quantize_int8_symmetric(a)
    wq, ws = nx.quantize_int8_symmetric(w)
    be = bridge.ReferenceBackend()
    got = be.matmul(aq, as_.reshape(-1), wq, ws.reshape(-1))
    expect = nx.int8_matmul(aq, as_, wq, ws)
    assert np.array_equal(got, expect)


# ---- the ctypes path, end to end against the stub --------------------
def test_stub_matches_the_cpp_reference_exactly(monkeypatch, stub_lib):
    """
    The core claim of this bridge: what ctypes marshals out and back is
    bit-identical to calling int8_gemm_ref directly in-process.
    """
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    be = bridge.get_backend()
    assert be.name == "cuda"    # dispatch treats the stub exactly like real CUDA

    rng = np.random.default_rng(1)
    M, N, K = 5, 7, 32   # K a multiple of 4, per the ABI's contract
    a_q = rng.integers(-127, 128, (M, K)).astype(np.int8)
    b_q = rng.integers(-127, 128, (N, K)).astype(np.int8)
    a_scale = rng.uniform(0.01, 1.0, M).astype(f32)
    b_scale = rng.uniform(0.01, 1.0, N).astype(f32)

    got = be.matmul(a_q, a_scale, b_q, b_scale)
    ref_expect = a_q.astype(np.int32) @ b_q.astype(np.int32).T
    ref_expect = ref_expect.astype(f32) * a_scale[:, None] * b_scale[None, :]
    assert np.array_equal(got, ref_expect)


def test_stub_rejects_k_not_a_multiple_of_4(monkeypatch, stub_lib):
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    be = bridge.get_backend()
    a_q = np.zeros((2, 5), np.int8)      # K=5, not a multiple of 4
    b_q = np.zeros((3, 5), np.int8)
    with pytest.raises(bridge.CudaGemmError):
        be.matmul(a_q, np.ones(2, f32), b_q, np.ones(3, f32))


def test_stub_handles_non_contiguous_input(monkeypatch, stub_lib):
    """Callers may hand in a transposed/sliced view; the bridge must copy to
    contiguous storage rather than misreading strided memory."""
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    be = bridge.get_backend()
    rng = np.random.default_rng(2)
    a_full = rng.integers(-127, 128, (8, 64)).astype(np.int8)
    a_q = a_full[::2]                     # non-contiguous view, shape (4, 64)
    assert not a_q.flags["C_CONTIGUOUS"]
    b_q = rng.integers(-127, 128, (3, 64)).astype(np.int8)
    a_scale = np.ones(4, f32)
    b_scale = np.ones(3, f32)

    got = be.matmul(a_q, a_scale, b_q, b_scale)
    expect = (a_q.astype(np.int32) @ b_q.astype(np.int32).T).astype(f32)
    assert np.array_equal(got, expect)


def test_stub_shape_mismatch_is_rejected_before_the_native_call(monkeypatch, stub_lib):
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    be = bridge.get_backend()
    a_q = np.zeros((2, 8), np.int8)
    b_q = np.zeros((3, 8), np.int8)
    with pytest.raises(ValueError, match="K mismatch"):
        be.matmul(a_q, np.ones(2, f32), np.zeros((3, 4), np.int8), np.ones(3, f32))
    with pytest.raises(ValueError, match="a_scale"):
        be.matmul(a_q, np.ones(5, f32), b_q, np.ones(3, f32))


# ---- the drop-in quantized_linear ------------------------------------
def test_quantized_linear_reference_backend_matches_numerics_exactly():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 128)).astype(f32)
    w = rng.standard_normal((9, 128)).astype(f32)
    got = bridge.quantized_linear(x, w, backend=bridge.ReferenceBackend())
    expect = nx.quantized_linear(x, w)
    assert np.array_equal(got, expect)


def test_quantized_linear_stub_cuda_matches_reference_backend(monkeypatch, stub_lib):
    """The claim that matters: swapping backends does not change the answer."""
    monkeypatch.setenv(bridge.ENV_OVERRIDE, str(stub_lib))
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 64)).astype(f32)
    w = rng.standard_normal((5, 64)).astype(f32)
    via_cuda = bridge.quantized_linear(x, w, backend=bridge.get_backend(force="cuda"))
    via_ref = bridge.quantized_linear(x, w, backend=bridge.ReferenceBackend())
    assert np.array_equal(via_cuda, via_ref)


def test_quantized_linear_defaults_to_get_backend(monkeypatch):
    """With nothing available, the default path still works end to end."""
    monkeypatch.delenv(bridge.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(bridge, "REPO_ROOT", bridge.REPO_ROOT / "nonexistent")
    x = np.ones((2, 32), f32)
    w = np.ones((3, 32), f32)
    out = bridge.quantized_linear(x, w)
    assert out.shape == (2, 3)
