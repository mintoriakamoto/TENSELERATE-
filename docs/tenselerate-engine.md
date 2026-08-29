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
    kvpool.py          paged KV blocks for the windowed full-attention layers
    scheduler.py       continuous batching: admit/retire every step
  gguf/                our own GGUF reader/writer (no llama.cpp dependency)
  server.py            OpenAI-compatible /v1 endpoint (stdlib, loopback-only)
  nvtx.py              NVTX markers (no-op without CUDA), per the spec directive
  backend/
    int8_gemm.py       ctypes bridge: CudaBackend / ReferenceBackend, get_backend()
  csrc/
    int8_gemm.h        int8 GEMM signature + portable C++ reference
    int8_gemm.cu       CUDA int8 GEMM (dp4a tiled) + the C ABI bridge wrapper
    int8_gemm_stub.cpp CPU stand-in for the bridge's C ABI (test fixture only)
    gemm_test.cpp      host test of the reference (runs with no GPU)
    CMakeLists.txt     host test always; CUDA lib + shared lib when a compiler is present
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

## One model only: Qwen3.8-27B

TENSELERATE serves exactly one model — the RavenX Chaos Agent (Qwen3.8-27B,
architecture `qwen3_5`). This is a hard limit, enforced at load time:
`config_from_gguf()` raises `UnsupportedModelError` for any file whose
architecture is not `qwen3_5` or whose geometry differs from the published 27B
config in any field. The engine's design assumes this one geometry everywhere —
the 16-of-64 attention split, the ~34 KiB/token KV rate, the window math, the
scheduler's batch caps — so "close enough" models are refused rather than served
with wrong numbers. `TINY` is a smoke-scale structural stand-in for tests and
the dev server only; it is never loadable from a GGUF file
(`tests/tenselerate/test_model_lock.py`).

## The 1,000,000-token floor

TENSELERATE never runs below **1,000,000 tokens** of context. `MIN_CONTEXT_TOKENS`
is a hard constant and `ModelConfig.validate_context()` refuses anything under it.

That floor has one unavoidable consequence, and it is the interesting part. The
model's trained rotary range is 262,144 tokens. Serving 1M with *full*
attention would extrapolate positions, which needs YaRN/RoPE scaling — and we do
not do RoPE scaling, because it costs quality. So the floor **forces** the
windowed hybrid:

- the **48 Gated-DeltaNet layers** carry long range in a fixed recurrent state
  and have **no positional encoding at all** — unbounded by construction,
  whether the sequence is 1M or 10M tokens;
- the **16 full-attention layers** attend a bounded `attention_window` that never
  exceeds the trained range, so no position is ever extrapolated.

`needs_rope_scaling()` therefore returns `False` at *any* context on a windowed
config, and a config with `attention_window=None` is refused at the floor rather
than silently extrapolating.

The window pays for itself twice. No position extrapolation, **and** a KV cache
whose size stops growing with context:

| context | KV (q8_0, 128K window) | RoPE scaling |
| --- | --- | --- |
| 1,000,000 | 4.25 GiB | no |
| 4,000,000 | 4.25 GiB | no |
| 10,000,000 | 4.25 GiB | no |

Decode speed follows KV, so it is **constant at any context at or above the
window**. That is the property that makes a 1M floor affordable at all.

### The window is the throughput dial

Each concurrent sequence carries its own windowed KV, so the window — not the
context — is what caps batch size, and batch size is what buys aggregate
throughput. On the unlocked CMP 170HX at the 1M floor:

| window | KV/seq | max concurrent | aggregate |
| --- | --- | --- | --- |
| 131,072 | 4.25 GiB | 11 | ~160 tok/s |
| 65,536 | 2.12 GiB | 22 | ~319 tok/s |
| **32,768** | **1.06 GiB** | **44** | **~638 tok/s** |
| 16,384 | 0.53 GiB | 88 | ~1,277 tok/s |

**Context is 1,000,000 in every row.** The window trades exact-recall depth (how
far back the full-attention layers see verbatim) for concurrency — never context
length, which the GDN state carries regardless. `tenselerate plan` computes this
table for the machine it is run on and refuses to print a batch size that would
not fit in VRAM.

Note the consumer box (24 GiB) fits the floor at a 128K window but only one
sequence; it wants a narrower window to get useful concurrency.

**The llama.cpp bridge cannot do this.** `scripts/tenselerate-serve.sh` runs the
stock model, whose full-attention layers are not windowed, so past 262,144 it
would need RoPE scaling. The bridge therefore caps at the model's native 256K;
the 1M floor is a property of the native engine.

## Continuous batching — the 600 tok/s lever

Decode is bandwidth-bound: one pass reads *all* the weights no matter how many
sequences are in flight. Running B sequences per step therefore costs barely more
than running one and yields B times the tokens. That is the entire reason 600+
tok/s is reachable on hardware whose single-stream ceiling is ~46.

Static batching throws it away, because the whole batch waits for its slowest
member. `tenselerate/engine/scheduler.py` admits and retires sequences **every
step**, so a finished sequence's slot is refilled immediately.

Two pools, matching the hybrid:

- **paged KV blocks** (`engine/kvpool.py`) for the 16 windowed full-attention
  layers — fixed-size, non-contiguous, so the pool cannot fragment;
- a **fixed per-sequence GDN state** for the 48 linear layers, which never grows.

**Admission is memory-first.** A sequence is admitted only when its *worst-case*
footprint — a full window of KV — is already available. That is deliberate:
admitting on current usage and hoping is how a server OOMs mid-generation, and a
sequence killed at token 400,000 of a 1,000,000-token context has wasted more work
than it ever produced. Over-admission raises `OutOfBlocks`; un-admittable work
raises a deadlock error rather than spinning forever.

The sliding window is what makes a 1M floor affordable here: a sequence's
blocks stop accumulating once it reaches the window, so `total_tokens` runs past
1,000,000 while `cached_tokens` holds flat and the pool never grows. That is pinned
by `test_kv_is_bounded_while_context_runs_past_the_floor`.

`tenselerate plan` asks the real `Scheduler` for its capacity rather than
recomputing it, so the throughput table above and the engine can never disagree.

## The int8 path, and the CMP

Decode leans on int8 symmetric-quantized matmuls accumulated in int32 — the IMMA
tensor-core path, which is **not** firmware-throttled on the CMP 170HX, unlike
dp4a dispatch and FP16 GEMM. `int8_gemm.h` carries the exact semantics; the first
CUDA kernel (`int8_gemm.cu`) is a portable **dp4a** implementation, correct and
validated, as the baseline. The CMP win comes from the `mma.sync` s8 IMMA kernel
(CC ≥ 7.5) that replaces it on that hardware — that is a roadmap item, and it will
be validated against the same reference before it ships.

## The kernel bridge

`tenselerate/backend/int8_gemm.py` is what turns a compiled CUDA kernel into
something the engine actually calls. It is a thin ctypes layer, not a compiled
Python extension, so importing the engine never needs a compile step - only
going fast does.

The C ABI is a host-pointer wrapper (`tenselerate_int8_gemm` in
`csrc/int8_gemm.cu`/`.h`) around the device-pointer kernel from the int8 GEMM
work: it owns the `cudaMalloc`/`Memcpy`/launch/`Memcpy`-back/`Free` sequence, so
the Python side only ever touches ordinary NumPy arrays. It is built into its
own `SHARED` CMake target (`tenselerate_int8_gemm_c`), separate from the static
library the CUDA-only tests link, because ctypes needs a `dlopen`-able `.so`.

`get_backend()` finds that library (searching this repo's `build-*` directories,
or an exact path via `TENSELERATE_INT8_GEMM_LIB`) and returns a `CudaBackend`;
with nothing found it returns the `ReferenceBackend` — the same NumPy path as
before — automatically. Nothing calling `quantized_linear()` needs to know which
one it got.

**Validated without a GPU.** `int8_gemm_stub.cpp` implements the identical C ABI
on the CPU (calls `int8_gemm_ref` instead of launching a kernel), compiled with
plain `g++`. Pointing the bridge at it with `TENSELERATE_INT8_GEMM_LIB` exercises
the *entire* marshaling path — pointer types, contiguity, error-code propagation
— through the exact mechanism that will load the real `.so`, with only the
computation swapped. `test_quantized_linear_stub_cuda_matches_reference_backend`
is the claim that matters: swapping backends does not change the answer.

## Roadmap (honest ordering)

| Phase | Item | Status |
| --- | --- | --- |
| 0 | Reference numerics + tests (int8 GEMM, GDN, RoPE, attention) | **done, CI-green** |
| 0 | Reference hybrid forward pass + decode loop | **done** |
| 0 | OpenAI `/v1` server (reference backend, byte tokenizer) | **done** |
| 0 | int8 GEMM CUDA kernel (dp4a) + CI compile for sm_80/86/120 | **done (compiles; runs on GPU only)** |
| 1 | GGUF reader (our own) + config-from-GGUF | **done, 9 tests** |
| 1 | 1M context floor + windowed hybrid attention (config + CLI) | **done, 10 tests** |
| 1 | `tenselerate` CLI (update/info/plan/doctor/serve) | **done, 9 tests** |
| 1 | Real tokenizer + weight load behind the server | not started |
| 1 | ctypes bridge so the engine calls the CUDA int8 GEMM | **done, 16 tests** |
| 1 | GDN linear-attention CUDA kernel (chunked scan) | not started |
| 1 | Flash-style causal attention kernel (16 full layers) | not started |
| 2 | Paged KV manager (full layers) + fixed GDN-state pool | **done, 16 tests** |
| 2 | Continuous batching scheduler — the 600 tok/s lever | **done, 16 tests** |
| 2 | IMMA `mma.sync` s8 GEMM — the CMP fast path | not started |
| 3 | MTP self-speculation (the built-in draft head) | not started |
| 3 | SVMI weight streaming for models that overflow VRAM | not started |

Phase 0 is a spine that builds, tests, and serves. Everything above phase 0 is
written against a reference that already exists, so each kernel lands with a
correctness bar on day one instead of being debugged blind.

## Running it now

```sh
# the CLI — one entry point for everything
python3 -m tenselerate info                    # geometry + the 1M floor
python3 -m tenselerate plan --machine cmp170hx # what this box does at the floor
python3 -m tenselerate doctor                  # driver/hardware check
python3 -m tenselerate update                  # check for a new build
python3 -m tenselerate update --source         # fast-forward and rebuild
python3 -m tenselerate serve --port 8080       # the OpenAI /v1 endpoint
```

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
