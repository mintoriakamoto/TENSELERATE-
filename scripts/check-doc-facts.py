#!/usr/bin/env python3
"""Guard the fork's docs against invented hardware facts and invented knobs.

Two independent checks, both scoped to markdown that talks about *this box*
(a file that never mentions the CMP 170HX or the TENSELERATE engine is out of
scope, so upstream llama.cpp docs are untouched and syncs stay conflict-free):

  1. CONTRADICTED CONSTANTS - a small table of hardware numbers this repo has
     already established by measurement. A doc asserting a different value for
     one of them is wrong, and the check says what the measured value is and
     where it came from.

  2. NONEXISTENT KNOBS - build flags, environment variables and binaries a doc
     tells the reader to use must exist somewhere in the tree. Instructions
     that cannot possibly work are the most expensive kind of wrong: the reader
     spends real time before finding out.

Check 2 is the general one. It catches fabrications nobody has seen yet, which
is the point - a denylist only ever catches the last mistake.

Usage:
    scripts/check-doc-facts.py [PATH ...]     # default: every *.md in the repo

Exit 0 when clean, 1 when a doc fails. Run by tests/tenselerate/test_doc_facts.py.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A file is in scope only if it discusses this box. Upstream docs do not.
IN_SCOPE = re.compile(r"170\s*HX|TENSELERATE|tenselerate", re.I)

# Fenced code blocks are where instructions live; prose may discuss a flag in
# the abstract ("upstream's -DGGML_FOO"), so knob checks read code blocks only.
FENCE = re.compile(r"^\s*```")


# Stay inside one sentence: any character except a newline or a sentence-ending
# period. A period that introduces a digit is a version or a decimal
# ("NVLink 3.0", "1.4 TB/s"), not a sentence break.
NEAR = r"(?:[^\n.]|\.(?=\d))"

# Saying the box has NO NVLink is the correct statement, so a negated mention
# is never a finding.
NEGATED_NVLINK = re.compile(r"\b(no|not|without|lacks|never|zero)\b[^\n]{0,20}?NVLink"
                            r"|NVLink[^\n]{0,20}?\b(absent|unavailable)\b", re.I)


@dataclass(frozen=True)
class Constant:
    """A measured fact, and the pattern that contradicts it."""
    name: str
    wrong: re.Pattern[str]
    truth: str
    source: str
    unless: re.Pattern[str] | None = None


# Every entry is a number this repo established by measurement, plus the doc
# that records it. Add a row when a doc starts asserting a new false constant.
CONSTANTS = (
    Constant(
        name="170HX memory bandwidth",
        wrong=re.compile(r"\b19[0-9]{2}\s*GB/s", re.I),
        truth="~1493 GB/s nominal (~890 GB/s achieved on the weight read)",
        source="docs/physics.md:8",
    ),
    Constant(
        name="170HX VRAM",
        wrong=re.compile(rf"170\s*HX{NEAR}{{0,60}}?\b80\s*G[Bi]B?\b|\b80\s*G[Bi]B?"
                         rf"{NEAR}{{0,40}}?HBM{NEAR}{{0,20}}?170\s*HX", re.I),
        truth="40 GiB unlocked (8 GiB stock; the --profile=8gb target is 64 GiB)",
        source="docs/rig-cmp170hx-3060.md:16",
    ),
    Constant(
        name="170HX host link",
        wrong=re.compile(rf"NVLink{NEAR}{{0,80}}?170\s*HX|"
                         rf"170\s*HX{NEAR}{{0,80}}?NVLink", re.I),
        truth="PCIe Gen2 x4, ~2 GB/s measured - there is no NVLink on this box, "
              "and that link is the constraint every design here works around",
        source="docs/rig-cmp170hx-3060.md:20",
        unless=NEGATED_NVLINK,
    ),
)

# Deliberately NOT a constant: "a single-stream tok/s above the 95 tok/s physics
# floor". Prose puts prefill and decode rates on one line ("pp4096 856 tok/s, tg
# 33.5 single stream"), and no pattern separated the impossible decode claim from
# the legitimate prefill one without flagging honest lines in README.md and
# CHANGELOG.md. A check that cries wolf gets ignored, which costs more than the
# bug it exists to catch - the knob check below is what catches invented numbers
# in practice anyway, because invented numbers travel with invented tooling.

# Exempt a line from the constant checks by ending it with this marker plus a
# reason - for the cases where quoting a wrong number is the point.
ALLOW = re.compile(r"<!--\s*doc-facts:allow\s+(.+?)\s*-->")


def code_block_lines(text: str) -> list[tuple[int, str]]:
    """Return (1-based line number, line) for lines inside fenced blocks."""
    out: list[tuple[int, str]] = []
    inside = False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            inside = not inside
            continue
        if inside:
            out.append((n, line))
    return out


# This file's whole purpose is to name flags, binaries and env vars that do NOT
# exist, so its contents must never count as evidence that one does. Without
# this the guard passes its own counterexamples: git ls-files lists tracked
# files, so the tests went green while unstaged and failed the moment they were
# committed. Any future fixture holding deliberate non-identifiers belongs here.
SELF_REFERENCE = ("tests/tenselerate/test_doc_facts.py",)


def _tree_text(patterns: tuple[str, ...]) -> str:
    """Concatenate every tracked file matching the globs. Cached per call site."""
    files = subprocess.run(
        ["git", "ls-files", "-z", *patterns],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    chunks = []
    for f in files:
        if not f or f in SELF_REFERENCE:
            continue
        try:
            chunks.append((ROOT / f).read_text(errors="replace"))
        except (OSError, IsADirectoryError):
            continue
    return "\n".join(chunks)


class Tree:
    """Lazily-read views of the repo, for 'does this identifier exist?'."""

    def __init__(self) -> None:
        self._cmake: str | None = None
        self._source: str | None = None

    @property
    def cmake(self) -> str:
        if self._cmake is None:
            self._cmake = _tree_text(("*CMakeLists.txt", "*.cmake", "*.yml", "*.sh"))
        return self._cmake

    @property
    def source(self) -> str:
        if self._source is None:
            self._source = _tree_text(
                ("*.c", "*.cpp", "*.cu", "*.cuh", "*.h", "*.hpp", "*.py",
                 "*.sh", "*.yml", "*CMakeLists.txt", "*.cmake"))
        return self._source


# CMake's own options are not defined in this tree; neither are a handful of
# third-party ones the build docs legitimately pass through.
CMAKE_BUILTIN = re.compile(r"^(CMAKE_|BUILD_SHARED_LIBS$|BUILD_TESTING$)")

CMAKE_FLAG = re.compile(r"-D([A-Z][A-Z0-9_]{2,})\s*[=\s]")
ENV_VAR = re.compile(r"\b((?:GGML|LLAMA)_[A-Z0-9_]{2,})\b")
BINARY = re.compile(r"(?:\./)?build[\w./-]*/bin/([a-z][a-z0-9-]{2,})")


def check_file(path: Path, tree: Tree) -> list[str]:
    text = path.read_text(errors="replace")
    if not IN_SCOPE.search(text):
        return []

    rel = path.relative_to(ROOT)
    problems: list[str] = []

    # 1. contradicted constants, over the whole document
    for n, line in enumerate(text.splitlines(), 1):
        if ALLOW.search(line):
            continue
        for c in CONSTANTS:
            if c.unless is not None and c.unless.search(line):
                continue
            if c.wrong.search(line):
                problems.append(
                    f"{rel}:{n}: {c.name} contradicts the measured value.\n"
                    f"    says     : {line.strip()[:110]}\n"
                    f"    measured : {c.truth}\n"
                    f"    source   : {c.source}")

    # 2. knobs that do not exist, in fenced code blocks only
    seen: set[str] = set()
    for n, line in code_block_lines(text):
        for flag in CMAKE_FLAG.findall(line):
            if flag in seen or CMAKE_BUILTIN.match(flag):
                continue
            seen.add(flag)
            if flag not in tree.cmake:
                problems.append(
                    f"{rel}:{n}: -D{flag} is not a build option in this tree "
                    f"(no CMakeLists.txt, *.cmake, workflow or script mentions it)")
        for var in ENV_VAR.findall(line):
            if var in seen:
                continue
            seen.add(var)
            if var not in tree.source:
                problems.append(
                    f"{rel}:{n}: ${var} is read by nothing in this tree")
        for binary in BINARY.findall(line):
            if binary in seen:
                continue
            seen.add(binary)
            if binary not in tree.cmake:
                problems.append(
                    f"{rel}:{n}: build/bin/{binary} is not a target in this tree")

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path,
                    help="markdown files to check (default: every tracked *.md)")
    args = ap.parse_args(argv)

    if args.paths:
        paths = [p if p.is_absolute() else ROOT / p for p in args.paths]
    else:
        listed = subprocess.run(["git", "ls-files", "-z", "*.md"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        paths = [ROOT / f for f in listed.stdout.split("\0") if f]

    tree = Tree()
    problems: list[str] = []
    checked = 0
    for p in paths:
        if not p.is_file():
            continue
        found = check_file(p, tree)
        if IN_SCOPE.search(p.read_text(errors="replace")):
            checked += 1
        problems.extend(found)

    if problems:
        print(f"{len(problems)} problem(s) in the docs that describe this box:\n")
        for p in problems:
            print(p)
        print("\nEach line above is either a measured constant a doc contradicts, "
              "or an instruction that cannot work.\nFix the doc. To quote a wrong "
              "number deliberately, end the line with "
              "'<!-- doc-facts:allow why -->'.")
        return 1

    print(f"doc facts: {checked} box-scoped doc(s) checked, no contradicted "
          f"constants and no nonexistent knobs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
