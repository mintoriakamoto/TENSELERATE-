# TENSELERATE (Cooklabs fork of llama.cpp)

This is a private first-party product fork (see [COOKLABS.md](COOKLABS.md)), not
upstream ggml-org/llama.cpp. The upstream contribution policy does not apply here.

## Code and Commit Standards

- Avoid emdash `-`, unicode arrows, or other unicode: use ASCII (`-`, `->`, `x`, `...`).
  The `editorconfig` CI check enforces this.
- Keep code comments concise; explain non-obvious invariants, not what the code already says.
- Prefer reusing existing infrastructure over introducing new subsystems.
- Read the relevant files before changing them; match the surrounding patterns.

## Useful Resources

- [Contributing guidelines](CONTRIBUTING.md)
- [SVMI docs](docs/svmi.md) and the [CMP 170HX + 3060 rig field guide](docs/rig-cmp170hx-3060.md)
- [Build documentation](docs/build.md)
- [Server usage](tools/server/README.md) and [server development](tools/server/README-dev.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
