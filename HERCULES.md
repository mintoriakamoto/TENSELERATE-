# Hercules — TENSELERATE

This tree is llama.cpp **plus** SVMI / stream-weights / INT8 / CMP-aware planners.
Product name: TENSELERATE.

Hercules consumes this via OpenAI-compat:

```bash
tenselerate boot --backend llamacpp --model MODEL.gguf   # doctor, then llama-server
#   or: bash scripts/hercules_serve.sh MODEL.gguf         (same launch, env overrides)
# then on the agent box:
hercules config set model.provider custom
hercules config set model.base_url http://127.0.0.1:8080/v1
```

The launch is built by `tenselerate/backends/llamacpp.py` from what the box
measured (`benches/cmp170hx-3060/`): 4 slots on a unified KV pool, no MTP,
`reasoning_effort=low`, chunked prefill with cache reuse. `--dry-run` prints
the command; `--no-mmvq` sets `GGML_CUDA_NO_MMVQ=1` once its runs confirm it.
Sizing and the slot table: `docs/rig-cmp170hx-3060.md`, "Serving Hercules".

Upstream is ggml-org/llama.cpp. Rebase from there. Do not drop the SVMI docs in `docs/svmi.md`.
