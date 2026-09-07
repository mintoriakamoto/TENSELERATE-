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
`reasoning_effort=low`, chunked prefill with cache reuse, and the flags Hermes
needs from an OpenAI-compatible server - `--jinja` (without it llama-server
ignores `tools` and the reasoning_effort template kwarg), `--reasoning-format
deepseek`, `--no-context-shift`, `--alias tenselerate`. `--dry-run` prints the
command; `--no-mmvq` sets `GGML_CUDA_NO_MMVQ=1` once its runs confirm it.
Sizing and the slot table: `docs/rig-cmp170hx-3060.md`, "Serving Hercules".

## Hermes side (`~/.hermes/config.yaml`)

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8080/v1
  default: tenselerate            # == --alias on the server
  context_length: 262144          # the locked window; Hermes compresses at 50% of this
  max_tokens: 8192
  streaming: true
providers:
  tenselerate:
    base_url: http://127.0.0.1:8080/v1
    models:
      tenselerate:
        context_length: 262144
        tool_parser: auto           # llama-server emits OpenAI tool_calls with --jinja
        reasoning_content: true     # keep <think> as reasoning_content, not prose
delegation:
  max_concurrent_children: 3      # v0.21 default is 10; the server is sized for 4 slots
compression:
  threshold: 0.50                 # the one sanctioned prompt-cache break; leave it
```

Why these: Hermes keeps the system prompt byte-stable and treats compression as
"the sanctioned cache break", so with the context declared correctly a session
re-prefills only when it compresses. Declared too small, it compresses (and
misses the prompt cache, ~70 s on a 60K context at 855 tok/s) far too early.
Hermes relaxes stream timeouts for local endpoints automatically; for very long
prefills set `HERMES_STREAM_READ_TIMEOUT=1800` in `.env`.

Upstream is ggml-org/llama.cpp. Rebase from there. Do not drop the SVMI docs in `docs/svmi.md`.
