# Hermes + TENSELERATE

Two engines on this PC. Do not mix ports.

| Engine | Tree | Port | Hermes provider |
|---|---|---|---|
| **TENSELERATE** (this fork) | `/home/ai/TENSELERATE` | **8083** | `tenselerate` / `hermes38-tenselerate` |
| llama-upstream (stock llama.cpp) | `/home/ai/llama-upstream` | **8082** | `llama-upstream` / `hermes38-upstream` |

Only one holds the 170HX at a time.

## Order of operations

1. NVIDIA driver (13.x UMD is OK).
2. CUDA **12.8** toolkit at `/usr/local/cuda-12.8` (not PATH `nvcc` 12.4).
3. `cmake --preset deploy-cmp170hx && cmake --build build-deploy-cmp170hx -j$(nproc) --target llama-server`
4. `bash scripts/boot-cmp170hx.sh` — waits for `/health`.
5. Hermes reads `~/.hermes/config.yaml` (symlink to `hermes-agent/config.yaml`):

```yaml
model:
  provider: tenselerate
  base_url: http://127.0.0.1:8083/v1
  default: hermes38-tenselerate
```

6. `hermes --yolo --tui` (or `tmux attach -t hermes`).

## Runtime flags (why)

| | |
|---|---|
| `GGML_CUDA_MMVQ_MAX=3` | 1-stream decode on MMVQ (~60 t/s). Width ≥4 uses MMQ. `NO_MMVQ=1` drops a single operator to ~34 t/s. |
| `-np 8 -c 262144 -kvu` | 8 Hermes workers, 256K **shared** KV. 8×64K dedicated and 512K OOM on 40 GB. |
| MTP n-max **4** | ~62 t/s, ~71% accept. n-max 8 OOM. |
| `--temp 0` | Draft head only pays under greedy. |
| CUDA 12.8 + `FORCE_MMQ` | Compile-time. Not a runtime switch. |

Hermes child agents are **new slots**. They do not reuse the parent’s 32k KV. Delegation knobs (`child_context_tokens`, `prefill_serialization`) live in Hermes config, not here.

## Health

```bash
curl -s http://127.0.0.1:8083/health
```
