# Fork development workflow (end to end)

This fork runs a standard GitHub flow: every change lands on `main` through
a pull request that CI has validated, and every native-code push to `main`
auto-publishes a `main-b<N>-<sha>` release with all platform artifacts.

## The loop

1. **Branch** off `main`:
   `git checkout -b feat/<short-name> origin/main`
2. **Build & test locally** before pushing:
   ```sh
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF
   cmake --build build -j
   ctest -L main --test-dir build --output-on-failure
   ```
   Python changes: `flake8` and `ty check` must be clean (CI enforces both).
   SVMI planner changes: run the matching `scripts/svmi-*.py --self-test`.
3. **Push and open a PR** against `main`. Fill in the template — the
   "How it was verified" checklist is the review.
4. **CI is the gate.** The PR triggers the same workflow fleet as `main`
   (cpu/apple/vulkan/webgpu/CUDA/server + lint/type-check). Nothing merges
   red; a flaky-looking failure gets one re-run, then investigation.
5. **Merge** via the GitHub UI/API — squash for small fixes, merge commit
   for multi-commit features. Delete the branch after merge.
6. **Release is automatic** on the `main` push. Scripts/docs-only pushes
   intentionally skip the heavy fleet (workflow path filters), so the
   released binaries always correspond to the last native-code commit.

## Branch protection (one-time, repo admin)

Settings → Branches → Add rule for `main`:

- Require a pull request before merging
- Require status checks to pass; suggested required set:
  `CI (cpu)`, `flake8 Lint`, `Python Type-Check`, `Code Style Checker`
  (add `CI (CUDA, ubuntu)` for release-critical native work)
- Require branches to be up to date before merging

Settings → General → enable **Automatically delete head branches**.

## Shipping updates to users

Two halves, and they meet at the release feed.

**Push (us).** Nothing extra to do: a native-code merge to `main` runs
`release.yml`, which publishes `main-b<N>-<sha>` with binaries for every
platform (ubuntu/macos/win x64+arm64, plus CUDA / ROCm / SYCL / Vulkan /
OpenVINO / Android variants). `<N>` is the commit count on `main`, so tags sort
the way builds do. Users who click **Watch -> Custom -> Releases** on the repo
get notified by GitHub the moment one lands - that is the push channel, and it
costs us nothing to maintain.

**Pull (them).** `scripts/tenselerate-update.sh` is the client:

```sh
scripts/tenselerate-update.sh --check    # exit 10 = update available
scripts/tenselerate-update.sh --source   # fast-forward main + rebuild in place
scripts/tenselerate-update.sh --binary   # download the prebuilt build instead
FLAVOR=cuda-12.4 scripts/tenselerate-update.sh --binary
scripts/tenselerate-update.sh --list     # every asset in the latest release
```

`--check` compares the local commit count against the `main-b<N>-<sha>` tag and
exits 10 when behind, so it drops into cron or a systemd timer:

```sh
30 4 * * * cd /opt/tenselerate && scripts/tenselerate-update.sh --check || \
    scripts/tenselerate-update.sh --source
```

**Users who only downloaded a binary** have no clone, so bootstrap the script
itself and let it read the version off the binary:

```sh
curl -fsSLO https://raw.githubusercontent.com/mintoriakamoto/TENSELERATE-/main/scripts/tenselerate-update.sh
chmod +x tenselerate-update.sh
./tenselerate-update.sh --check      # reads `llama-cli --version`
./tenselerate-update.sh --binary     # replaces it with the current release
```

It looks for the binary at `$LLAMA_BIN`, `build/bin/llama-cli`, an unpacked
`dist/*/llama-cli`, or `llama-cli` on `PATH`, and reports `running: b<N> (sha)`
from `--version`. In a clone it prints `running` *and* `source`, and says so when
the checkout is ahead of the binary you are actually executing - that gap is
otherwise invisible and looks like "the update did nothing".

`--source` refuses to run on a dirty tree, fast-forwards only (never a merge
commit), and reuses the existing `build/CMakeCache.txt` so a CMP box keeps its
`-DGGML_CUDA_DISABLE_DP4A=ON -DGGML_CUDA_FORCE_MMQ=ON` configuration across
updates instead of silently rebuilding without them. `--binary` picks the plain
build for the machine and asks for a `FLAVOR` when several match. Point the
script at a different fork with `TENSELERATE_REPO=owner/name`; it otherwise
reads the clone's own `origin`.

## Ground rules from this tree's history

- **Formats are upstream-canonical**: QK2_0 = 64, gated-delta-net snapshot
  slot 0 = most recent. Anything assuming the PrismML fork's 128-weight
  Q2_0 layout must be gated off per backend (see the CUDA/Vulkan/Metal
  `supports_op` gates) — a wrong-result kernel is worse than no kernel.
- **Backends must reject what they can't run** so ggml falls back to CPU.
- **Claims need validators**: new SVMI methods ship a `--self-test`
  (`scripts/svmi-distspec.py` et al.) or a ctest; docs record the numbers
  the validator produced on commit day.
- Commit messages explain *why*; PR descriptions carry verification
  evidence.
