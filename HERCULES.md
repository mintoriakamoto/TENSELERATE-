# Hercules — TENSELERATE

This tree is llama.cpp **plus** SVMI / stream-weights / INT8 / CMP-aware planners.
Product name: TENSELERATE.

Hercules consumes this via OpenAI-compat:

```bash
tenselerate boot --backend llamacpp --model MODEL.gguf   # doctor, then llama-server
#   or: bash scripts/hercules_serve.sh MODEL.gguf         (same launch, env overrides)
# then on the agent box:
hercules config set model.provider custom
hercules config set model.base_url http://127.0.0.1:8083/v1
```

Production on this 170HX is `scripts/boot-cmp170hx.sh` (** :8083 **, 8 slots,
256K unified, MTP n-max 4, `GGML_CUDA_MMVQ_MAX=3`). Hercules can use the same
server, or `scripts/hercules_serve.sh` with:

```bash
NP=8 CTX=262144 PORT=8083 MMVQ_MAX=3 MTP=4 ALIAS=hermes38-tenselerate \
  bash scripts/hercules_serve.sh MODEL.gguf
```

`--no-mmvq` is **not** for a single operator (−34% vs `MMVQ_MAX=3`). Greedy
(`--temp 0`) is required for MTP to pay. `--jinja` is required for Hermes tools.
Sizing: `docs/rig-cmp170hx-3060.md`. Hermes wiring: `docs/hermes.md`.

## Hermes side (`~/.hermes/config.yaml`)

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8083/v1
  default: hermes38-tenselerate            # == --alias on the server
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

## The system prompt is prefilled once, not once per agent

Hermes sends the same ~35K-token system prompt with the main loop and with
every delegation child. At 855 tok/s that is ~40 s of prefill each time a
slot has to build it from scratch. The launch turns on the three server
features that stop that:

| flag | what it does here |
| --- | --- |
| `--cache-ram 16384` | a host-RAM prompt cache (MiB). When a slot goes idle its KV is saved here and, with `--kv-unified`, the slot is cleared. 16 GiB holds ~7 copies of the 35K prefix at q8_0 |
| `--cache-idle-slots` | the save-on-idle above, made explicit (it is on by default when the cache is) |
| `--slot-prompt-similarity 0.1` | a request goes to the idle slot whose cached prompt shares the longest common prefix with it; otherwise to the LRU slot, which then loads the best-matching prefix from the RAM cache. A child arriving while the parent's slot is busy gets the 35K prefix from RAM instead of re-prefilling it |
| `--slot-save-path DIR` (opt-in, `SLOT_SAVE_PATH=`) | exposes `/slots/<id>?action=save\|restore`. After the first turn, `bash scripts/hercules_slots.sh save`; after a restart, `bash scripts/hercules_slots.sh restore` before the first request. The prefix then survives the restart |

Verify in the server log: a child's request should show `selected slot by LCP
similarity` or a `prompt cache` load with `n_past` near 35K, not a full
prefill.

## Delegation without transport: `fork_from`

A delegated sub-agent normally starts by rebuilding the parent's context: a
prompt-cache load (~0.6 s for the 35K prefix over this card's Gen2 x4 link) or
a full prefill (~40 s). Neither is necessary. A sequence on this model is KV
cells for the attention layers plus one recurrent state cell for the 48 GDN
layers, and both can be shared on the card by reference.

Pass the parent's slot id in the completion request:

```json
{ "prompt": "...", "fork_from": 0 }
```

The server points a free slot at the parent's cells and state, clones the token
list, and prefills only the delta. The child inherits the parent's **whole**
context, including the associative memory of everything that has already fallen
out of the attention window - which a prompt-cache load cannot give it, because
the cache only carries the prefix the child was sent.

Requirements and fallbacks: the source slot must be idle and its cached context
must be a prefix of the new request. Anything else (busy source, unknown id, no
free slot, divergent prefix) logs one line and falls back to normal slot
selection, so it is safe to always send. Slot ids come from `GET /slots`.

Verified bit-exact by `tests/test-seq-fork.cpp`: a forked sequence's logits
equal an independently decoded one, and the parent is unaffected by the child's
decoding.

### Keeping a system-prompt template: pin its slot

The same mechanism gives every agent a warm system prompt: dedicate one slot to
the shared prefix and have new conversations fork from it, instead of each one
prefilling or loading it from the cache. With
`--slot-prompt-similarity > 0` and `LLAMA_SERVER_SLOT_FORK=1` the server even
finds that donor on its own, so clients need not pass `fork_from` at all.

**Pin that slot, or it will be the first one destroyed.** Slot selection - both
the LRU fallback and the fork-target search - picks the idle slot with the
oldest `t_last_used`, and a template that only ever donates never updates its
timestamp. It is the oldest slot in the server by construction. It survives
until the first request that does not share its prefix, and then every agent
quietly goes back to paying the full prefill.

```sh
LLAMA_SERVER_PIN_SLOTS=0 LLAMA_SERVER_SLOT_FORK=1 \
  tenselerate boot --backend llamacpp --model MODEL.gguf
```

Pinned slots donate context and are never scheduled, so slot 0 keeps the prompt
for the life of the server. Warm it once after start with a request carrying
`"id_slot": 0`. Pinning every slot is refused at startup rather than accepting a
request that can never be served.

## The Hermes knobs that actually move speed on a local server

From Hermes' configuration reference (v0.21). Defaults in parentheses; the
right-hand column is what they cost or save on this box.

| key | default | set to / why |
| --- | --- | --- |
| `delegation.max_concurrent_children` | 3 in the feature doc, 10-16 in newer references - **set it explicitly** | 3: the server has 4 slots; a 4th+ child queues |
| `delegation.max_iterations` | 50 (500 in newer refs) | leave; each child turn is a full decode pass |
| `delegation.child_timeout_seconds` | 0 (none) | set e.g. 900 so a stuck child frees its slot |
| `delegation.max_spawn_depth` | 1 | leave at 1; orchestrator children multiply slots |
| `auxiliary.compression.model` / `.base_url` | main model | same second server: the 50% compaction summary then does not occupy a 170HX slot or the 27B's 30 ms/token |
| `auxiliary.title_generation.enabled` | true | false: one fewer model call per session |
| `compression.threshold` | 0.50 | leave; every compaction is a prompt-cache miss (~70 s at 60K), so fewer is better, not lower |
| `compression.proactive_prune_tokens` | 0 (off) | leave off unless contexts bloat; pruning rewrites earlier messages = cache miss |
| `tool_output.max_bytes` / `max_lines` | 50000 / 2000 | 20000 / 800: tool results are what grow the context toward compaction |
| **system prompt size** (toolsets, skills, memory) | Hermes ships ~35K tokens of system prompt with every toolset enabled | `agent.disabled_toolsets` for anything Hercules does not use, trim skills, keep `memory.memory_char_limit` / `user_char_limit` at defaults: 35K is ~40 s of prefill on every cache miss and ~2.3 GB of KV per slot at q8_0 |
| `terminal.timeout` | 180 s | raise for long builds/tests; this is tool time, not model time |
| `terminal.backend`, `container_cpu`, `container_memory` | local, 1, 5120 | if sandboxing in Docker, give it real cores (4-8) - tool latency is wall-clock on the agent loop |
| `agent.max_turns` | none | cap runaway loops (e.g. 60) |
| `model.reasoning_effort` | unset | **honored since the upstream sync**: the server maps a request's `reasoning_effort` into the template (`none` disables thinking for that request). The launch's `--reasoning-effort low` is the default; set this per model in Hermes only if you want a different level, and use `none` for tool-heavy turns that need no thinking |
| **`temperature` / `repeat_penalty` in requests** | Hermes sends none by default | **leave them unset.** llama.cpp accepts a drafted token only if the sampled token equals it; a request that carries temp 0.7 / repeat-penalty 1.15 overrides the server's greedy defaults and drops MTP from 46.2 to 29.9 tok/s (below no-MTP). Verify in the server log: `draft acceptance` ~0.88 during a Hermes turn |
| `model.streaming` | true | leave on |
| `HERMES_STREAM_READ_TIMEOUT` | 120 s (1800 auto for local) | 1800 explicitly if deep prefills trip it |

Why these: Hermes keeps the system prompt byte-stable and treats compression as
"the sanctioned cache break", so with the context declared correctly a session
re-prefills only when it compresses. Declared too small, it compresses (and
misses the prompt cache, ~70 s on a 60K context at 855 tok/s) far too early.
Hermes relaxes stream timeouts for local endpoints automatically; for very long
prefills set `HERMES_STREAM_READ_TIMEOUT=1800` in `.env`.

Upstream is ggml-org/llama.cpp. Rebase from there. Do not drop the SVMI docs in `docs/svmi.md`.
