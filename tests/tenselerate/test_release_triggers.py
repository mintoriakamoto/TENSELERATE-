"""A release must publish whenever the binaries could have changed.

The Release workflow builds llama-server, llama-cli and the int8 kernel bridge,
and `scripts/tenselerate-update.sh --binary` installs what it publishes. Its
paths-ignore list therefore has an asymmetric risk: ignoring one path too few
wastes a 33-minute build, while ignoring one too many means a real code change
publishes nothing and the box silently keeps running a stale binary.

These tests pin the safe direction. Every path the compiler can see must still
trigger a release; only paths with a stated reason may be ignored.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / ".github" / "workflows" / "release.yml"


ITEM = re.compile(r"^\s+- '([^']+)'\s*$")


def ignore_patterns() -> list[str]:
    """Read the paths-ignore list out of release.yml.

    Parsed directly rather than with PyYAML, which is not a dependency of this
    repo and would be one more thing to install for one test. The block is a
    flat list of quoted strings; anything else fails the assertions below
    loudly rather than silently returning an empty list, which would make every
    test here pass for the wrong reason.
    """
    lines = RELEASE.read_text().splitlines()
    start = [i for i, ln in enumerate(lines) if ln.strip() == "paths-ignore:"]
    assert len(start) == 1, "expected exactly one paths-ignore block"

    out = []
    for ln in lines[start[0] + 1:]:
        if ln.strip().startswith("#") or not ln.strip():
            continue
        m = ITEM.match(ln)
        if not m:
            break
        out.append(m.group(1))

    assert out, "parsed no ignore patterns - has release.yml changed shape?"
    assert "**.md" in out, "the markdown rule vanished; re-check the parser"
    return out


def to_regex(pattern: str) -> re.Pattern[str]:
    """GitHub filter glob: ** crosses separators, * does not."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def publishes(changed: list[str]) -> bool:
    """GitHub runs the workflow if ANY changed path escapes every ignore rule."""
    rules = [to_regex(p) for p in ignore_patterns()]
    return any(not any(r.match(f) for r in rules) for f in changed)


# Anything the release build compiles or links. If a change here stops
# publishing, the updater serves a binary that does not contain it.
@pytest.mark.parametrize("path", [
    "src/llama-model.cpp",
    "src/tenselerate-kv-mean-center.cpp",
    "ggml/src/ggml-cuda/mmvq.cu",
    "ggml/src/ggml-cuda/fattn-vec.cuh",
    "common/arg.cpp",
    "common/kv-mean-center.cpp",
    "tools/server/server-context.cpp",
    "tenselerate/csrc/int8_gemm.cu",
    "CMakeLists.txt",
    "cmake/build-info.cmake",
    "tests/test-seq-fork.cpp",
    ".github/workflows/release.yml",
    ".github/actions/package-release.sh",
    "scripts/build-info.sh",
])
def test_native_paths_still_publish(path):
    assert publishes([path]), (
        f"{path} would no longer publish a release. If that is intended, say "
        f"why the compiler never sees this file - otherwise the box keeps "
        f"running a binary without this change.")


@pytest.mark.parametrize("path", [
    "README.md",
    "docs/physics.md",
    "benches/cmp170hx-3060/README.md",
    "tests/tenselerate/test_doc_facts.py",
    "scripts/check-doc-facts.py",
    ".github/workflows/doc-facts.yml",
])
def test_documentation_and_python_tooling_do_not_publish(path):
    assert not publishes([path]), f"{path} should not trigger a 33-minute build"


def test_the_doc_facts_pr_would_not_have_published():
    # PR #76 rebuilt 1.14 GB of CUDA binaries that could not have differed by a
    # single byte: it changed only Python tooling, tests and CI definitions.
    assert not publishes([
        ".github/workflows/doc-facts.yml",
        ".github/workflows/tenselerate-engine.yml",
        "AGENTS.md",
        "docs/upstream-sync.md",
        "scripts/check-doc-facts.py",
        "tests/tenselerate/test_doc_facts.py",
    ])


def test_one_code_file_in_a_docs_push_is_enough_to_publish():
    # the rule is per-push, not per-file: a mixed push must still publish
    assert publishes(["README.md", "docs/physics.md", "ggml/src/ggml-cuda/mmvq.cu"])


def test_scripts_are_not_ignored_wholesale():
    # scripts/build-info.sh feeds the version cmake stamps into the binary, so
    # a blanket 'scripts/**' would be wrong; only named files may be ignored
    assert not any(p in ("scripts/**", "scripts/*", "tests/**")
                   for p in ignore_patterns())


# --------------------------------------------------------------------------
# The published binary's build configuration.
#
# Two properties of the CUDA build are load-bearing on this box and neither is
# visible in the tarball's name, so they are pinned here rather than left to a
# reviewer to notice:
#
#   CUDA 12.8.1  the runtime is linked statically, so the toolkit the release
#                container carries IS the runtime that executes on the cards.
#                A bare "12.8" tag would float across patch levels.
#   FORCE_MMQ    without it ggml_cuda_should_use_mmq() falls through to cuBLAS
#                once the batch is wide enough (sm_80 has fp16 mma), so prefill
#                and other wide-batch matmuls leave the int8 MMQ path and pay a
#                dequant to fp16. This is a routing change, not a measured win:
#                the number belongs in benches/cmp170hx-3060/README.md.
#
# CI must compile against the same toolkit or it is not a pre-merge proxy for
# what ships, so the engine workflow is held to the same image.
ENGINE = ROOT / ".github" / "workflows" / "tenselerate-engine.yml"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"


def test_release_builds_cuda_12_8_1_with_forced_mmq():
    body = RELEASE.read_text()
    assert f'CUDA_IMAGE: "{CUDA_IMAGE}"' in body, "release.yml must pin the CUDA patch level"
    assert "container: ${{ env.CUDA_IMAGE }}" in body, "the CUDA job must use the pinned image"
    assert "-DGGML_CUDA_FORCE_MMQ=ON" in body, "the published binary must force the MMQ path"
    # the flag is worthless if it never reaches the artifact, so the build asserts
    # on llama-cli's own feature line
    assert 'grep -q "FORCE_MMQ"' in body, "release.yml must verify FORCE_MMQ in the built binary"


def test_ci_compiles_against_the_same_toolkit_as_the_release():
    body = ENGINE.read_text()
    assert f'CUDA_IMAGE: "{CUDA_IMAGE}"' in body, "CI must compile against the release toolkit"
    assert "container: ${{ env.CUDA_IMAGE }}" in body
    assert "-DGGML_CUDA_FORCE_MMQ=ON" in body, "CI must compile the same MMQ routing as the release"


def test_the_option_the_workflows_pass_actually_exists():
    # a renamed upstream option would make both workflows silently no-ops
    opts = (ROOT / "ggml" / "CMakeLists.txt").read_text()
    assert re.search(r"^option\(GGML_CUDA_FORCE_MMQ\b", opts, re.M), \
        "GGML_CUDA_FORCE_MMQ is not an option in ggml/CMakeLists.txt any more"
    cuda_cmake = (ROOT / "ggml" / "src" / "ggml-cuda" / "CMakeLists.txt").read_text()
    assert "add_compile_definitions(GGML_CUDA_FORCE_MMQ)" in cuda_cmake, \
        "the option no longer turns into a compile definition"
