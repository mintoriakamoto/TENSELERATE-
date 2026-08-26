# TENSELERATE — native inference engine

Our own way to host models. Not a llama.cpp fork, not a wrapper around vLLM or
ExLlama — a from-scratch engine whose compute path we own end to end, built for
the hardware we actually run: the CMP 170HX (no usable FP16, int8 tensor cores
intact) and consumer GeForce, serving the `qwen3_5` hybrid.

This document is the map and the honest status. It is deliberately blunt about
what is real, what is a stub, and what is not written yet.

## Why from scratch

The [`scripts/tenselerate-serve.sh`](../scripts/tenselerate-serve.sh) path over
the llama.cpp fork works **today** and is the right thing to run right now. The
native engine is the long game: full control of the KV manager, the scheduler,
and the kernels, so nothing about the CMP's throttled dp4a / dead FP16 / narrow
PCIe is someone else's assumption to fight. vLLM and ExLlamaV3 both run this
model well on FP16-healthy cards; neither is tuned for the CMP, and neither is
ours to change at the kernel level. That is the gap this engine fills.

## Architecture

```
tenselerate/
  config.py            ModelConfig; RAVENX_27B mirrors the real config.json,
                       TINY is the same STRUCTURE at smoke scale
  reference/
    numerics.py        pure-NumPy reference for every hot op (the oracle)
    model.py           reference forward pass of the hybrid (proves the wiring)
  engine/
    generation.py      the decode loop (no blocking calls in the hot path)
  server.py            OpenAI-compatible /v1 endpoint (stdlib, loopback-only)
  nvtx.py              NVTX markers (no-op without CUDA), per the spec directive
  csrc/
    int8_gemm.h        int8 GEMM signature + portable C++ reference
    int8_gemm.cu       CUDA int8 GEMM (dp4a tiled; IMMA is the next step)
    gemm_test.cpp      host test of the reference (runs with no GPU)
    CMakeLists.txt     host test always; CUDA lib when a compiler is present
tests/tenselerate/     correctness tests, all CPU, all green in CI
```

**The reference is the source of truth.** Every kernel has a NumPy or C++
reference that runs with no GPU, and the tests pin the math. A CUDA kernel is
correct when it matches its reference; until then the engine runs the reference
so the whole pipeline — config, model, decode loop, server — is exercised end to
end on any machine. This is why `python -m tenselerate.server` already answers
Hermes's `/v1/chat/completions` today (with meaningless tokens: random weights,
byte tokenizer — the *contract* is real, the *weights* are not yet).

## The hybrid, and why it drives every design choice

`qwen3_5` is 64 layers: **16 full-attention, 48 Gated-DeltaNet linear**. The
full layers grow a KV cache; the linear layers keep a fixed-size recurrent state
(`tests/.../test_gdn_state_is_fixed_size_regardless_of_length`). That single fact
is why a 24 GiB box reaches the model's native 256K context, and it shapes the KV
manager: two pools, one growing (paged, for the 16 full layers) and one fixed
(per-sequence GDN state), sized so they share physical memory cleanly.

The GDN recurrence is validated in two forms — sequential and chunked — that must
agree (`test_gdn_chunked_equals_sequential`). The chunked form is exactly what the
CUDA kernel parallelizes over, so that test is the kernel's spec.

## The int8 path, and the CMP

Decode leans on int8 symmetric-quantized matmuls accumulated in int32 — the IMMA
tensor-core path, which is **not** firmware-throttled on the CMP 170HX, unlike
dp4a dispatch and FP16 GEMM. `int8_gemm.h` carries the exact semantics; the first
CUDA kernel (`int8_gemm.cu`) is a portable **dp4a** implementation, correct and
validated, as the baseline. The CMP win comes from the `mma.sync` s8 IMMA kernel
(CC ≥ 7.5) that replaces it on that hardware — that is a roadmap item, and it will
be validated against the same reference before it ships.

## Roadmap (honest ordering)

| Phase | Item | Status |
| --- | --- | --- |
| 0 | Reference numerics + tests (int8 GEMM, GDN, RoPE, attention) | **done, CI-green** |
| 0 | Reference hybrid forward pass + decode loop | **done** |
| 0 | OpenAI `/v1` server (reference backend, byte tokenizer) | **done** |
| 0 | int8 GEMM CUDA kernel (dp4a) + CI compile for sm_80/86/120 | **done (compiles; runs on GPU only)** |
| 1 | GGUF weight loader + real tokenizer behind the same server | not started |
| 1 | ctypes/pybind bridge so the engine calls the CUDA int8 GEMM | not started |
| 1 | GDN linear-attention CUDA kernel (chunked scan) | not started |
| 1 | Flash-style causal attention kernel (16 full layers) | not started |
| 2 | Paged KV manager (full layers) + fixed GDN-state pool | not started |
| 2 | Continuous batching scheduler (steal vLLM's idea, our code) | not started |
| 2 | IMMA `mma.sync` s8 GEMM — the CMP fast path | not started |
| 3 | MTP self-speculation (the built-in draft head) | not started |
| 3 | SVMI weight streaming for models that overflow VRAM | not started |

Phase 0 is a spine that builds, tests, and serves. Everything above phase 0 is
written against a reference that already exists, so each kernel lands with a
correctness bar on day one instead of being debugged blind.

## Running it now

```sh
# reference tests (no GPU)
PYTHONPATH=. pytest tests/tenselerate/ -q

# dev server — the real /v1 contract, reference backend
PYTHONPATH=. python3 -m tenselerate.server --port 8080
curl -s localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}'

# host int8 GEMM reference test
cmake -S tenselerate/csrc -B build-kernels && cmake --build build-kernels
ctest --test-dir build-kernels --output-on-failure

# CUDA kernels (needs nvcc >= 12.8 for sm_120)
cmake -S tenselerate/csrc -B build-cuda -DCMAKE_CUDA_ARCHITECTURES="80-real;86-real;120-real"
cmake --build build-cuda
```
