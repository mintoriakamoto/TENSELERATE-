## What

<!-- one paragraph: what does this PR change and why -->

## How it was verified

- [ ] `cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build -j` compiles clean
- [ ] `ctest -L main --test-dir build` passes (or failures explained below)
- [ ] Python touched? `flake8` and `ty check` are clean
- [ ] SVMI planners touched? relevant `scripts/svmi-*.py --self-test` passes

## Risk / compatibility notes

<!-- quant format changes, backend gating, GGUF/API compatibility, release impact -->
