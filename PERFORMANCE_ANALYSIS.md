# TENSELERATE production flags (CMP 170HX + Hermes)

This box: one CMP 170HX (GA100, sm_80), 40 GB, 250 W. TENSELERATE is its own
engine on **:8083**. `llama-upstream` (`ggml-org/llama.cpp`) on **:8082** is a
different binary. Do not mix providers.

Measured on this binary (CUDA 12.8 + FORCE_MMQ, Qwen3.8-27B TurboFCFusion Q4_K_M):

| | |
|---|---|
| Prefill ~8k tokens | **~737 tok/s** |
| Decode (greedy, MTP n-max 4, MMVQ_MAX=3) | **~60–62 tok/s** |
| llama-bench pp4096 (on-card record) | **856 tok/s** |
| 6500 t/s | **not this card** (Blackwell 5090 / 4B Spark tables) |

## Build (compile-time)

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --preset deploy-cmp170hx
cmake --build build-deploy-cmp170hx -j$(nproc) --target llama-server
MODEL=/path/to/model.gguf bash scripts/boot-cmp170hx.sh
```

`deploy-cmp170hx` inherits `cuda-int8` + `cmp-dp4a`:

| CMake | Value | Why |
|---|---|---|
| `CMAKE_CUDA_COMPILER` | `/usr/local/cuda-12.8/bin/nvcc` | PATH nvcc is 12.4; 12.8 is the MMQ toolkit |
| `CMAKE_BUILD_RPATH` | `/usr/local/cuda-12.8/lib64` | load 12.8 cudart, not distro 12.4 |
| `CMAKE_CUDA_ARCHITECTURES` | `80-real` | this card only; no JIT |
| `GGML_CUDA` | ON | required |
| `GGML_CUDA_FORCE_MMQ` | **ON** | integer MMQ (via `cuda-int8`) |
| `GGML_CUDA_DISABLE_DP4A` | **ON** | native `__dp4a` on Ampere CMP is ~16× slow |
| `GGML_CUDA_FORCE_CUBLAS` | **OFF** | cuBLAS is the slow path on this stack |
| `GGML_CUDA_GRAPHS` | ON | keep |
| `GGML_NATIVE` | OFF | GPU-only |

Driver UMD 13.3 is fine. Compiling **with** CUDA 13 toolkit is the trap.

## Runtime (boot)

`scripts/boot-cmp170hx.sh` (Hermes wrapper: `~/.hermes/scripts/hermes-boot-tenselerate.sh`).

| Flag | Value | Why |
|---|---|---|
| `GGML_CUDA_MMVQ_MAX` | **3** | 1-stream stays MMVQ (~60 t/s). Width 4+ goes MMQ. **Do not** set `NO_MMVQ=1` for a single Hermes operator (that was 34 t/s). |
| `LD_LIBRARY_PATH` | `/usr/local/cuda-12.8/lib64` | 12.8 runtime |
| `numactl --membind=0` | yes | NUMA |
| `-np` | **8** | Hermes workers |
| `-c` | **262144** `-kvu` | 256K **shared** pool (8×64K dedicated and 512K OOM) |
| `-b` / `-ub` | 8192 / 2048 | wide prefill |
| `-ctk/-ctv` | q8_0 | KV |
| `-fa` | on | flash attention |
| MTP | `draft-mtp` **n-max 4** p-min 0 | n-max 3 ≈ 60 t/s 81% acc; n-max 4 ≈ 62 t/s 71% acc; n-max 8 OOM |
| `--temp` | 0 | greedy; sampling kills MTP |
| port | **8083** | not 8082 |
| model | TurboFCFusion **Q4_K_M ~18G** | |

## Do not

- `FORCE_CUBLAS=ON`
- `GGML_CUDA_NO_MMVQ=1` on 1-stream Hermes (use `MMVQ_MAX=3`)
- Treat 6500 t/s as a 170HX 27B number
- Point the tenselerate provider at :8082
- Boot 512K / n-max 8 first (OOM every restart)
