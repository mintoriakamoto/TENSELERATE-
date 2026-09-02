# Cooklabs — TENSELERATE

This tree is llama.cpp **plus** SVMI / stream-weights / INT8 / CMP-aware planners.
Product name: TENSELERATE. Company map: https://github.com/mintoriakamoto/Cooklabs

Hercules consumes this via OpenAI-compat:

```bash
./build/bin/llama-server -m MODEL.gguf --port 8080
# then on the agent box:
hercules config set model.provider custom
hercules config set model.base_url http://127.0.0.1:8080/v1
```

Planner: `python3 scripts/svmi-plan.py MODEL.gguf --gpu 3060`

Upstream is ggml-org/llama.cpp. Rebase from there. Do not drop the SVMI docs in `docs/svmi.md`.
