"""The docs that describe this box must not invent hardware or knobs.

A doc that states the wrong bandwidth sends someone down a modeling path that
cannot pay off; a doc that names a build flag or a benchmark binary that does
not exist costs the reader an afternoon before they find out. Both are cheap to
catch mechanically, so they are caught mechanically.

scripts/check-doc-facts.py holds the checks; this pins them to CI and pins the
behaviour that keeps the guard usable - it only looks at docs about this box,
so upstream llama.cpp docs never trip it and syncs stay conflict-free.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check-doc-facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_doc_facts", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # the module must be registered before exec: @dataclass resolves its
    # annotations through sys.modules[cls.__module__]
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def check(tmp_path: Path, name: str, body: str) -> list[str]:
    """Run the checks over one throwaway doc placed inside the repo tree."""
    doc = ROOT / f"_doc_facts_probe_{name}.md"
    doc.write_text(body)
    try:
        return mod.check_file(doc, mod.Tree())
    finally:
        doc.unlink()


def test_the_repo_is_clean():
    rc = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert "no contradicted constants" in rc.stdout


def test_a_new_untracked_doc_is_still_checked(tmp_path):
    """A doc just written must be checked, not silently skipped until committed.

    git ls-files lists tracked files only. Checking just those means a
    contributor writes a doc, runs this, sees it pass, commits, and CI fails on
    the very file they were adding - the same tracked/untracked blind spot that
    made the guard pass its own counterexamples.
    """
    doc = ROOT / "_doc_facts_untracked_probe.md"
    doc.write_text("# 170HX\n\nThe CMP 170HX runs HBM2e at 1935 GB/s.\n")
    try:
        listed = mod._git_files(("*.md",))
        assert doc.name in listed, "an untracked markdown file was not listed"
        rc = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                            capture_output=True, text=True)
        assert rc.returncode == 1, "the untracked doc's false constant was missed"
        assert "bandwidth" in rc.stdout
    finally:
        doc.unlink()


def test_a_doc_that_never_mentions_the_box_is_out_of_scope(tmp_path):
    # this is what keeps upstream's docs/build.md from ever failing this check
    body = "# Build\n\n```bash\ncmake -B build -DGGML_INVENTED=ON\n```\n"
    assert check(tmp_path, "scope", body) == []


@pytest.mark.parametrize("line,expect", [
    ("The CMP 170HX runs HBM2e at 1935 GB/s.", "bandwidth"),
    ("CMP 170HX leverages 80GB for staging.", "VRAM"),
    ("NVLink 3.0 links the CMP 170HX pair.", "host link"),
])
def test_contradicted_constants_are_caught(tmp_path, line, expect):
    found = check(tmp_path, "const", f"# Notes\n\n{line}\n")
    assert len(found) == 1, found
    assert expect in found[0]
    # the failure has to be actionable: it names the real value and its source
    assert "measured :" in found[0] and "source   :" in found[0]


def test_the_true_constants_pass(tmp_path):
    body = ("# Notes\n\nThe CMP 170HX has 40 GiB unlocked and ~1493 GB/s "
            "nominal, on PCIe Gen2 x4 at ~2 GB/s. Batch 1 measures 33.5 tok/s "
            "against a 95 tok/s floor; N=32 aggregates 134 tok/s.\n")
    assert check(tmp_path, "true", body) == []


def test_throughput_numbers_are_never_flagged(tmp_path):
    # there is no tok/s rule: a line can carry a prefill rate, a decode rate and
    # an aggregate at once, and no pattern told the impossible claim from the
    # honest ones without flagging README.md. See the note in the script.
    body = ("# Notes\n\nTENSELERATE: pp4096 856 tok/s, tg 33.5 single stream, "
            "342 tok/s aggregate across 8 slots.\n")
    assert check(tmp_path, "rates", body) == []


def test_a_negated_nvlink_mention_is_correct_not_a_finding(tmp_path):
    body = "# Notes\n\nThe CMP 170HX box has no NVLink; it is PCIe Gen2 x4.\n"
    assert check(tmp_path, "neg", body) == []


def test_nonexistent_build_flag_is_caught(tmp_path):
    body = ("# 170HX\n\n```bash\ncmake -B build -DENABLE_NVLINK_OPTIMIZATION=ON\n```\n")
    found = check(tmp_path, "flag", body)
    assert len(found) == 1 and "ENABLE_NVLINK_OPTIMIZATION" in found[0]


def test_real_build_flag_passes(tmp_path):
    body = "# 170HX\n\n```bash\ncmake -B build -DGGML_CUDA=ON\n```\n"
    assert check(tmp_path, "realflag", body) == []


def test_nonexistent_binary_is_caught(tmp_path):
    body = "# 170HX\n\n```bash\n./build/bin/benchmark-gqa-attention --tokens=262144\n```\n"
    found = check(tmp_path, "bin", body)
    assert len(found) == 1 and "benchmark-gqa-attention" in found[0]


def test_real_binary_passes(tmp_path):
    body = "# 170HX\n\n```bash\n./build/bin/llama-bench -m model.gguf\n```\n"
    assert check(tmp_path, "realbin", body) == []


def test_nonexistent_env_var_is_caught(tmp_path):
    body = "# 170HX\n\n```bash\nexport GGML_CUDA_MAKE_IT_FAST=1\n```\n"
    found = check(tmp_path, "env", body)
    assert len(found) == 1 and "GGML_CUDA_MAKE_IT_FAST" in found[0]


def test_this_file_is_not_evidence_that_anything_exists(tmp_path):
    """The guard must not read its own counterexamples as real identifiers.

    git ls-files lists tracked files only, so before this file was committed
    the invented names above were invisible and every test passed; committing
    them made GGML_CUDA_MAKE_IT_FAST "exist" and turned CI red. The exclusion
    is what makes the result the same either way.
    """
    assert "tests/tenselerate/test_doc_facts.py" in mod.SELF_REFERENCE
    tree = mod.Tree()
    for invented in ("GGML_CUDA_MAKE_IT_FAST", "ENABLE_NVLINK_OPTIMIZATION",
                     "benchmark-gqa-attention"):
        assert invented in Path(__file__).read_text(), invented
        assert invented not in tree.source, invented
        assert invented not in tree.cmake, invented


def test_real_env_var_passes(tmp_path):
    # the fork's own knobs, which must keep working after every upstream sync
    body = ("# 170HX\n\n```bash\nexport GGML_CUDA_MMVQ_MAX=3\n"
            "export LLAMA_ATTN_WINDOW=32768\nexport GGML_CUDA_FATTN_VEC_GQA=1\n```\n")
    assert check(tmp_path, "realenv", body) == []


def test_prose_is_not_read_for_knobs(tmp_path):
    # prose discusses flags that may belong to other projects; only fenced
    # instructions are held to "this must exist here"
    body = "# 170HX\n\nvLLM builds with -DVLLM_TARGET_DEVICE=cuda, unlike us.\n"
    assert check(tmp_path, "prose", body) == []


def test_an_allow_marker_exempts_a_line(tmp_path):
    body = ("# 170HX\n\nA stale note claimed 1935 GB/s, which was never true. "
            "<!-- doc-facts:allow quoting the error being corrected -->\n")
    assert check(tmp_path, "allow", body) == []
