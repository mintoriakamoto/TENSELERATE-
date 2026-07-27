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
