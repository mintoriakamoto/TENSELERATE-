# Syncing the fork to upstream llama.cpp

The fork is not a git fork. Its history starts with an "Import upstream
ggml-org/llama.cpp master snapshot" commit whose tree is byte-identical to
upstream `64c8b7db72fbd871512b371b5c141c00fd0a8ba6` (2026-07-09, "server :
respect min-step when splitting prompt batches"). Because there is no shared
commit, `git merge upstream/master` refuses ("unrelated histories") and a
forced merge with an empty base conflicts on every file. The right merge is a
three-way merge with that snapshot as the explicit base.

## What upstream carries since the snapshot (915 commits as of 2026-09-07)

The ones that move this box, from `docs/research-week-2026-09-07.md`:
- #26705 MMVQ +13-16% on the dp4a vector path (every single-slot decode step here)
- #27991 KV-cache restore ~100x faster (slot save/restore, prompt-cache misses)
- DSpark / DFlash2 drafters that fail in server mode on the fork's
  `common/speculative.cpp` (July 23 vintage)
- #28208, #25173, #27342 (see the weekly scan)

## Procedure (ran to the conflict list on 2026-09-07)

```
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch --unshallow origin            # the clone must not be shallow
git fetch --unshallow upstream master
git diff --stat cd062118a 64c8b7db7      # must print nothing: snapshot == upstream base
git checkout -b sync-upstream
git read-tree -m -u --aggressive 64c8b7db7 HEAD upstream/master
git merge-index -o git-merge-one-file -a
git ls-files -u | awk '{print $4}' | sort -u   # the conflict list
```

Result: **73 conflicted paths.**

- 29 are `.github/workflows/*.yml` that the fork deleted and upstream edited.
  Resolution: keep the deletion (`git rm`). The fork runs its own two
  workflows (`tenselerate-engine.yml`, `release.yml`).
- `ggml/src/ggml-metal/ggml-metal.metal`: upstream deleted it (metal moved to
  per-op files). Take the deletion.
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_0.comp`: both sides added a
  Q2_0 shader. Take upstream's, and check that the fork's Q2_0 additions in
  `ggml/include/ggml.h`, `gguf-py/gguf/constants.py`, `vecdotq.cuh`,
  `dequantize.cuh`, `mmq.cuh` agree with upstream's enum values - if upstream
  also added a Q2_0 the fork's must be dropped in favour of upstream's numbering.
- 42 real content conflicts. By area:
  - speculative decoding: `common/speculative.{cpp,h}`, `common/arg.cpp`,
    `common/common.h`, `examples/speculative-simple/`, `tools/server/server-context.cpp`
    (the fork's draft-dspark / MTP plumbing vs upstream's rewritten drafters -
    prefer upstream, then re-apply only what upstream lacks)
  - the fork's quant/kernel work: `ggml-cuda/{mmvq.cu,mmq.cuh,vecdotq.cuh,dequantize.cuh}`,
    `template-instances/generate_cu_files.py`, `ggml-cpu/repack.cpp`, vulkan
    dequant shaders, `ggml-metal-*`, `ggml-sycl.cpp`, `ggml/src/CMakeLists.txt`,
    `ggml-backend.cpp`. `mmvq.cu` also carries this branch's
    `GGML_CUDA_MMVQ_MAX` threshold (three dispatch sites; see `mmvq.cuh`).
  - model code: `src/llama-{arch,model,graph,context,kv-cache,memory-recurrent,mmap}.cpp`,
    `src/llama-ext.h` (the fork's svmi / weight-streaming hooks)
  - `AGENTS.md`, `README.md`, `tests/CMakeLists.txt`, `scripts/hip/gcn-cdna-vgpr-check.py`

## Verification before it lands

1. CPU build: `cmake -B build-cpu -DGGML_CUDA=OFF && cmake --build build-cpu -j` and
   `ctest --test-dir build-cpu`.
2. CI: the engine workflow compiles `ggml-cuda` for sm_80 (the only place the
   CUDA side is checked without the card).
3. On the box: `llama-bench -p 4096 -n 64`, then the MTP depth-1 greedy run.
   Upstream's MMVQ change should show up directly in tg64 (33.5 -> ~38).
4. `python3 -m tenselerate serve --backend llamacpp --dry-run` still passes
   the flag set (`--spec-type draft-mtp`, `--kv-unified`, `--cache-reuse`).

## Status: landed 2026-09-07 (three-way merge, upstream 67672dc5)

Resolution actually taken, by area:

- **CUDA / Vulkan / Metal / SYCL / CPU quant kernels: upstream.** Upstream
  carries Q1_0 and Q2_0 at the same enum values as the fork, so the fork's
  kernel-side additions for them were superseded rather than lost. Dropped
  with that: the fork's CPU repack fast path for Q1_0/Q2_0 (`arch/*/repack.cpp`)
  and its Metal Q1_0 routing knobs. Neither is on this box's path.
- **`GGML_CUDA_MMVQ_MAX` kept.** `mmvq.cu` merges the fork's threshold check
  ahead of upstream's new per-architecture MMVQ tuning table (Ada, Blackwell,
  DGX Spark, Orin); the two dispatch sites in `ggml-cuda.cu` auto-merged.
- **Speculative decoding: upstream.** `common/speculative.{cpp,h}`, the
  speculative-simple example and `tools/server/server-context.cpp` are
  upstream's. The fork's own dspark drafter (multi-layer capture staging,
  `common_speculative_need_embd_capture`, two manual test harnesses) is gone;
  upstream's `draft-dspark` (DFlash + Markov head) and `draft-dflash` take
  its place and work in server mode. The fork's `LLM_ARCH_DSPARK` model,
  capture-layer API (`llama_set_capture_layers`), `test-dspark-forward`,
  and `llama-ext.h` extensions stay. The server keeps one fork fix: the
  `slot_batched->is_processing()` guard before `llama_set_embeddings`.
- **Fork-only features kept as ours:** `ggml_gated_delta_net_rows` (ggml.h +
  CPU op), the recurrent-state snapshot ring (`rs_ring`, all three
  `llama-memory-recurrent.cpp` hunks), the mmap host-pin unregister in
  `~llama_mmap` (combined with upstream's lazy-range constructor),
  kv-mean-center (tool, common, tests), `test-rs-ring-rotation`, README,
  AGENTS.md, this repository's two workflows and `release.yml`.
- **Deleted:** the 29 upstream workflow files the fork had already removed,
  `ggml-metal.metal` (upstream split it), `tests/test-dspark-loop.cpp`,
  `tests/test-dspark-real-eval.cpp`.

Verification: CPU build (`-DGGML_CUDA=OFF`, tools + server + tests) - see the
commit message for the result; CI compiles `ggml-cuda` for sm_80; the
tenselerate suite (152) and the launch dry run pass on the merged tree.
Every flag the launch emits exists in upstream's `common/arg.cpp`.

What the box should see first: `llama-bench -p 4096 -n 64` (upstream's MMVQ
work: tg64 33.5 -> ~38 predicted), then the greedy MTP depth-1 run, then
`--spec-type draft-dspark` with a DimInfer/RadixArk head, which now runs in
server mode.
