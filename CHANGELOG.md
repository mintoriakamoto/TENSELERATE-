# Changelog

Releases are cut automatically from every native-code push to `main` as
`main-b<N>-<sha>` (assets: `bin-ubuntu-x64` CPU, `bin-ubuntu-cuda-12.8-sm80-86-x64`
static CUDA). This file groups them by what changed. Upstream llama.cpp changes
arrive through the sync PRs and are not repeated here.

## Unreleased

- **Production 170HX boot** (`scripts/boot-cmp170hx.sh`, `cmake --preset deploy-cmp170hx`):
  CUDA 12.8, `FORCE_MMQ=ON`, `DISABLE_DP4A=ON`, `GGML_CUDA_MMVQ_MAX=3`,
  8×256K unified, MTP n-max 4, port 8083. Hermes/Hercules wiring in
  `docs/hermes.md` and `HERCULES.md`. Live: ~737 t/s prefill, ~60 t/s decode.

- **Ampere back-ports of Ada+ scheduling**, opt-in and off by default:
  `GGML_CUDA_FATTN_ADA_GATE=1` takes the Ada+ flash-attention kernel choice on
  Ampere (with quantized KV, Ada and newer keep batch width 2 - an MTP depth-1
  verify pass - on the vector kernel, where Ampere drops to MMA), and
  `GGML_CUDA_FATTN_STREAM_K=1|0` overrides the "Ada+ or tile efficiency < 75%"
  stream-k heuristic. Both are scheduling, not instructions, and run on sm_80.
  Survey of what is and is not portable from sm_90/sm_120 in
  `docs/ampere-backports.md`.
- **CI: the CUDA compile now runs on any `ggml/src/ggml-cuda/**` change.** The
  path filter had listed only `mmvq.*` and `ggml-cuda.cu`, so seven fork-touched
  CUDA files - the GQA-packed attention among them - were skipping the
  sm_80/sm_86 compile.

- **On-card sequence fork** (`"fork_from": <slot id>` on a completion request).
  A delegated agent starts from a live slot's state by sharing its KV cells and
  its recurrent state cell - `llama_memory_seq_cp` plus copy-on-write in VRAM -
  instead of a prompt-cache load (~0.6 s over this card's Gen2 x4 link) or a
  full prefill (~40 s for the 35K Hermes prefix). The child inherits the
  parent's whole context including the GDN memory past the attention window.
  Falls back to normal slot selection whenever the fork does not apply. Pinned
  bit-exact by `tests/test-seq-fork.cpp`.
- Checked the parallel "fork GDN is corrupted" diagnosis against main: all four
  claimed divergences are stale or misread (`benches/cmp170hx-3060/README.md`),
  and added `fork-vs-upstream-ab.sh` to settle the reported numbers by
  measurement.

- **Bounded attention window for the GDN hybrid** (`LLAMA_ATTN_WINDOW`,
  `LLAMA_ATTN_SINKS`; `tenselerate serve --attn-window --attn-sinks`). The
  16 full-attention layers of Qwen3.5/3.8 become sliding-window layers over
  hybrid-iswa memory with pinned leading positions; the 48 GDN layers carry
  the long range. KV per slot is O(window), sequences run past the training
  context, decode cost is flat in depth. Off by default; identity below the
  window is tested bit-exact (`test-attn-window`). Design and predictions in
  `docs/bounded-window-serving.md`.

## main-b11044-073223f - 2026-09-08

- **Fix: upstream merge leftovers** (#52). Recurrent state, hybrid memory,
  delta-net, qwen35, graph, context, model and arch sources are upstream's
  again; the fork's snapshot ring had been left mixed with upstream's
  rollback, which broke partial `seq_rm` on GDN models (the MTP verify crop).
  Metal back to upstream. Fork-only dspark model, capture API and ring test
  removed; upstream's `draft-dspark` (dflash) is what the server uses.
  kv-mean-center hooks re-applied. All recurrent rollback tests pass.

## main-b11040-031b6ef - 2026-09-08

- `--reasoning-effort` is the first-class thinking control emitted by the
  launch builder; per-request `reasoning_effort: none` disables thinking (#50).
- GQA-packed vector attention auto gate raised to 32K KV (#50).
- llama-bench experiments in the bench runner: regression bisect (packed
  attention off/on, old binary), prefill micro-batch sweep, quant A/B (#51).

## main-b11038-28b819e - 2026-09-08

- **Prompt-cache serving flags** (#49): `--cache-ram` (host RAM prompt cache),
  `--cache-idle-slots`, `--slot-prompt-similarity` (LCP slot selection),
  `--slot-save-path` with `scripts/hercules_slots.sh save|restore|erase`.
  Prefill the Hermes system prompt once, serve it many times.
- Sampling guards for MTP: `--sampling greedy|dry|low|client`; `dry` and
  `low` stop the `<think>` loops greedy decoding falls into on this merge.
- Side-model launcher for the RTX 3060 (`scripts/hercules_side_serve.sh`).
- Release workflow: CUDA build version check runs against the libcuda stub (#48).

## main-b11036-4f8f50e - 2026-09-07

- **Upstream sync**: 915 upstream commits merged by three-way merge on the
  true base (the fork was a snapshot with no git ancestry), 73 conflicts
  resolved. Brings upstream MMVQ speedups, 100x faster KV restore, Q1_0/Q2_0,
  `draft-dspark`/`draft-dflash` in server mode. Procedure and resolution
  record in `docs/upstream-sync.md` (#47).
- **GQA-packed vector flash attention** for quantized K/V at single-token
  decode (`fattn-vec.cuh`, `GGML_CUDA_FATTN_VEC_GQA`).
- Measured CMP 170HX + RTX 3060 results and corrected rig guidance:
  pp4096 856 tok/s, tg 33.5 single stream, 48.2 aggregate at 2x256K,
  70.5 aggregate at 4x256K with `GGML_CUDA_NO_MMVQ=1`; the decode cost
  model (18.5 ms weight read + 11.5 ms per extra dp4a row) and the MMQ
  55 ms floor (`benches/cmp170hx-3060/README.md`).
- MTP tooling: depth-1 greedy 46.2 tok/s at 0.88 acceptance; sampled
  acceptance analysis; `scripts/mtp-head-train.py` Route A trainer.
- `docs/kernel-work.md`: the three open CUDA items with the arithmetic.

## main-b138-f43b644 and earlier - 2026-08 to 2026-09-07

- Model switch to DavidAU Qwen3.8-27B TURBO Q4_K_M (#46); 256K window at
  max recall (#45); Ampere retarget sm_80;sm_86 (#43, #44).
- vLLM backend option and the Ampere box plan (#42).
- Reference oracle correctness fixes (KV eviction, float64 RoPE angle,
  GDN geometry) and engine-only CI (#31).
- SVMI: unlocked CMP 170HX presets, 3060 + 170HX box planner, rig field
  guide (#38); Turing-acceleration research roadmap (#36).
- INT8 mixed quant (`Q4_K_M_INT8`), all-integer CUDA build presets, weight
  streaming (`--stream-weights`, `--stream-decode`), pinned host store,
  planners (`scripts/svmi-*.py`), update channel (`scripts/tenselerate-update.sh`).
