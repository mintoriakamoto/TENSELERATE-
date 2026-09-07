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
measured (`benches/cmp170hx-3060/`): 4 slots on a unified KV pool, MTP at
draft depth 1 (`--mtp-draft 1`, +35% measured on this merge; deeper loses),
**greedy sampling** (`--temp 0 --repeat-penalty 1.0` as server defaults: the
draft head pays only under greedy - 46.2 tok/s at 88% acceptance vs 29.9 at
22% with the model card's temp 0.7 / repeat-penalty 1.15, which is slower than
no MTP at all), `reasoning_effort=low`, chunked prefill with cache reuse, and the flags Hermes
needs from an OpenAI-compatible server - `--jinja` (without it llama-server
ignores `tools` and the reasoning_effort template kwarg), `--reasoning-format
deepseek`, `--no-context-shift`, `--alias tenselerate`. `--dry-run` prints the
command; `--mmvq-max 3` sets the fork's
`GGML_CUDA_MMVQ_MAX=3`: single-slot turns and depth-1 verification stay on the
dp4a path (which wins below width ~4), four-slot steps go to the tensor cores
(+19% measured). `--no-mmvq` (all widths to MMQ) measured -34% on single-slot
MTP - do not use it for one operator.
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

## The Hermes knobs that actually move speed on a local server

From Hermes' configuration reference (v0.21). Defaults in parentheses; the
right-hand column is what they cost or save on this box.

| key | default | set to / why |
| --- | --- | --- |
| `delegation.max_concurrent_children` | 3 in the feature doc, 10-16 in newer references - **set it explicitly** | 3: the server has 4 slots; a 4th+ child queues |
| `delegation.max_iterations` | 50 (500 in newer refs) | leave; each child turn is a full decode pass |
| `delegation.child_timeout_seconds` | 0 (none) | set e.g. 900 so a stuck child frees its slot |
| `delegation.max_spawn_depth` | 1 | leave at 1; orchestrator children multiply slots |
| `delegation.model` / `.base_url` | inherit | **route children to a second, smaller server on the RTX 3060** (below) |
| `auxiliary.compression.model` / `.base_url` | main model | same second server: the 50% compaction summary then does not occupy a 170HX slot or the 27B's 30 ms/token |
| `auxiliary.title_generation.enabled` | true | false: one fewer model call per session |
| `compression.threshold` | 0.50 | leave; every compaction is a prompt-cache miss (~70 s at 60K), so fewer is better, not lower |
| `compression.proactive_prune_tokens` | 0 (off) | leave off unless contexts bloat; pruning rewrites earlier messages = cache miss |
| `tool_output.max_bytes` / `max_lines` | 50000 / 2000 | 20000 / 800: tool results are what grow the context toward compaction |
| **system prompt size** (toolsets, skills, memory) | Hermes ships ~35K tokens of system prompt with every toolset enabled | `agent.disabled_toolsets` for anything Hercules does not use, trim skills, keep `memory.memory_char_limit` / `user_char_limit` at defaults: 35K is ~40 s of prefill on every cache miss and ~2.3 GB of KV per slot at q8_0 |
| `terminal.timeout` | 180 s | raise for long builds/tests; this is tool time, not model time |
| `terminal.backend`, `container_cpu`, `container_memory` | local, 1, 5120 | if sandboxing in Docker, give it real cores (4-8) - tool latency is wall-clock on the agent loop |
| `agent.max_turns` | none | cap runaway loops (e.g. 60) |
| `model.reasoning_effort` | unset | llama-server does **not** map this request field into the chat template; keep `--chat-template-kwargs reasoning_effort` on the server side (the launch does) |
| **`temperature` / `repeat_penalty` in requests** | Hermes sends none by default | **leave them unset.** llama.cpp accepts a drafted token only if the sampled token equals it; a request that carries temp 0.7 / repeat-penalty 1.15 overrides the server's greedy defaults and drops MTP from 46.2 to 29.9 tok/s (below no-MTP). Verify in the server log: `draft acceptance` ~0.88 during a Hermes turn |
| `model.streaming` | true | leave on |
| `HERMES_STREAM_READ_TIMEOUT` | 120 s (1800 auto for local) | 1800 explicitly if deep prefills trip it |

### Two servers, two cards

The RTX 3060 has been a spectator. Its 12 GiB and 360 GB/s serve a ~9B model
(Qwen3.5-9B Q4_K_M, ~5.5 GiB + KV) at roughly 40-55 tok/s single stream -
faster per token than the 27B - and Hermes routes children and the compaction
summarizer there with `delegation.base_url` / `auxiliary.compression.base_url`.
The 170HX then holds only the main loop: one deep slot, f16 KV, the full
window. Whether a 9B child is good enough for your subtasks is a quality call;
the speed case is that it takes subagent and summary traffic off the 27B
entirely.

```
CUDA_VISIBLE_DEVICES=<3060> llama-server -m qwen3.5-9b-Q4_K_M.gguf --alias side \
  --host 127.0.0.1 --port 8081 --jinja --reasoning-format deepseek --no-context-shift \
  -ngl 999 -fa on -c 131072 -np 4 --kv-unified -cb -ctk q8_0 -ctv q8_0 \
  --chat-template-kwargs '{"reasoning_effort":"low"}'
```

```yaml
delegation:
  base_url: http://127.0.0.1:8081/v1
  model: side
auxiliary:
  compression:
    base_url: http://127.0.0.1:8081/v1
    model: side
```

Why these: Hermes keeps the system prompt byte-stable and treats compression as
"the sanctioned cache break", so with the context declared correctly a session
re-prefills only when it compresses. Declared too small, it compresses (and
misses the prompt cache, ~70 s on a 60K context at 855 tok/s) far too early.
Hermes relaxes stream timeouts for local endpoints automatically; for very long
prefills set `HERMES_STREAM_READ_TIMEOUT=1800` in `.env`.

Upstream is ggml-org/llama.cpp. Rebase from there. Do not drop the SVMI docs in `docs/svmi.md`.
