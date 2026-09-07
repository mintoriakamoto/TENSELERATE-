# Hercules — TENSELERATE

This tree is llama.cpp **plus** SVMI / stream-weights / INT8 / CMP-aware planners.
Product name: TENSELERATE.

Hercules consumes this via OpenAI-compat:

```bash
bash scripts/hercules_serve.sh MODEL.gguf     # measured config for the 170HX + 3060 box
# then on the agent box:
hercules config set model.provider custom
hercules config set model.base_url http://127.0.0.1:8080/v1
```

The serve script encodes what was measured on the box (`benches/cmp170hx-3060/`):
4 slots on a unified KV pool, no MTP, `reasoning_effort=low`. Sizing and the
slot table: `docs/rig-cmp170hx-3060.md`, "Serving Hercules".

Upstream is ggml-org/llama.cpp. Rebase from there. Do not drop the SVMI docs in `docs/svmi.md`.
