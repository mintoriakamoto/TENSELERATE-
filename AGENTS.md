# TENSELERATE - working notes for contributors and coding agents

This is a maintained fork of `ggml-org/llama.cpp`. Read this before changing
anything; `CONTRIBUTING.md` points here for the AI-usage policy.

## What this repository is

- `src/`, `ggml/`, `common/`, `tools/`, `examples/`: llama.cpp, kept close to
  upstream. Fork-owned changes are listed in `docs/upstream-sync.md` and are
  the only places to expect divergence.
- `tenselerate/`: the Python CLI (`tenselerate boot|serve|doctor|plan|...`),
  the llama.cpp launch builder (`backends/llamacpp.py`), the reference oracle
  (`reference/`) and its numerics tests.
- `scripts/`: SVMI planners (`svmi-*.py`), Hercules serving scripts
  (`hercules_*.sh`), the update client (`tenselerate-update.sh`), MTP tooling.
- `benches/cmp170hx-3060/`: the measured record for the reference box and the
  runner for the open experiments. Predictions are written before runs and
  graded after; keep that order.
- `docs/`: `status-*.md` (current state), `kernel-work.md` (open CUDA items),
  `ROADMAP.md`, `upstream-sync.md`, `dev-workflow.md`, `HERCULES.md` at root.

## Build and test

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF
cmake --build build -j
ctest -L main --test-dir build --output-on-failure
python3 -m pytest tests/tenselerate -q          # ~5 s
flake8 tenselerate tests/tenselerate            # CI enforces flake8 and ty
```

CUDA: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86"
-DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_DISABLE_DP4A=ON`, built against **CUDA 12.8.1** - the release links the
CUDA runtime statically, so the toolkit that builds it is the one that runs on
the cards, and the patch level is pinned (`CUDA_IMAGE` in both workflows, held
by `tests/tenselerate/test_release_triggers.py`). `FORCE_MMQ` keeps quantized
matmuls on the int8 MMQ path instead of falling through to cuBLAS once the
batch is wide; the release build asserts it reached the binary by grepping
`llama-cli --version`. `DISABLE_DP4A` swaps `__dp4a` for a `prmt`+`dp2a`
emulation: the CMP 170HX dispatches dp4a about 16x slower than regular silicon,
so the fork's own kernel comment records ~2x end-to-end decode from the swap. It
is the fork's option, it is reported in `--version` by a one-line
`TENSELERATE` hook so the build can assert it, and it would be the wrong flag on
a non-CMP card. CI compiles the CUDA backend for sm_80 on every PR
against the same image; there is no GPU in CI, so kernel changes are proven on
the box with the release binary.

Tests that matter for the fork's own code: `test-recurrent-state-rollback*`
(GDN rollback, the MTP verify crop), `test-kv-mean-center`,
`test-save-load-state`, `test-backend-ops`, and the tenselerate suite.

## Rules

1. **Upstream first, and stay out of upstream's files.** If upstream has the
   feature, take upstream's version and delete the fork's. Fork code goes in
   fork-owned files (`src/tenselerate-*.cpp`, `common/kv-mean-center.*`);
   an upstream source gets at most a one-line hook, marked with a
   `TENSELERATE` comment. `scripts/fork-hunks.sh` lists the footprint; that
   list is what conflicts on every upstream sync, so keep it short.
   Fork-only code needs a test and an entry in `docs/upstream-sync.md`.
2. **Measure on the release binary.** Performance claims go in
   `benches/cmp170hx-3060/README.md` as measured rows, with the prediction
   they grade. A dev-build number is a note, not a result.
3. **Every flag the launch emits must exist in `common/arg.cpp`.** Run the
   dry run (`tenselerate boot --dry-run` or the tenselerate tests) after
   touching the launch builder or syncing upstream.
4. **PRs, not direct pushes.** Branch from `main`, fill the PR template's
   verification checklist, CI green, then merge. The `main` push publishes
   the release.
5. **Do not skip or quarantine tests to get green.** Fix or revert.
6. **Docs about this box state measured values and working instructions.**
   The card is 40 GiB unlocked at ~1493 GB/s on PCIe Gen2 x4 (~2 GB/s), with
   no NVLink; those are the numbers every design here works around, so a doc
   that contradicts them sends the reader down a path that cannot pay off. A
   build flag, `GGML_*`/`LLAMA_*` env var or `build/bin/` binary shown in a
   fenced block must exist in the tree. `scripts/check-doc-facts.py` enforces
   both on any markdown that mentions the 170HX or TENSELERATE (upstream docs
   are out of scope, so syncs stay clean). To quote a wrong number on purpose,
   end the line with `<!-- doc-facts:allow why -->`.
7. **AI-assisted changes are allowed and must be disclosed** in the commit or
   PR. The author is responsible for every line.

## Conventions

- Commit subjects: `area: what changed` (`cuda:`, `llamacpp:`, `benches:`,
  `docs:`, `release:`). Body says why and how it was verified.
- Python: type-annotated, `flake8` and `ty` clean, no new dependencies
  without a note in `pyproject.toml`.
- C++/CUDA: match upstream style (`.clang-format`), env knobs documented in
  the README feature table, defaults off unless measured.
- Docs: numbers carry a date and the binary they were measured on.
- Releases: every native-code push to `main` publishes ~1.15 GB of assets and
  the updater only ever resolves `/releases/latest`, so `release.yml` prunes to
  the newest 8 of this fork's own releases. It never touches anything else;
  `scripts/prune-releases.sh` (dry run by default) is the human-run tool for
  the rest. Do not name a release tag in a doc or script - it will be pruned.
