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

## Status

Not landed. The merge was carried to the conflict list in this repository's
working tree and abandoned there: the session that ran it is not permitted to
delete the 29 workflow files, which is the first resolution step. Everything
above is reproducible from the commands; the 42 content conflicts are a
day's work with a build in the loop.
