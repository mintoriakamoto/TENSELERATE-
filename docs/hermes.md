# Hermes + TENSELERATE

TENSELERATE (this fork) is the model server. Hermes is a separate agent. They talk over OpenAI-compat on **:8083**.

If you also run stock `ggml-org/llama.cpp`, use another port (this box uses **:8082**). One GPU: only one server loaded.

## Order of operations

1. NVIDIA driver (13.x UMD is OK).
2. CUDA **12.8** toolkit at `/usr/local/cuda-12.8`. Not 12.4. Not 13.x toolkit.
3. Clone, `cmake --preset deploy-cmp170hx`, build `llama-server`.
4. `bash scripts/fetch-model.sh` then `bash scripts/boot-cmp170hx.sh` — exact GGUF `Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`.
5. Hermes `~/.hermes/config.yaml`:

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
