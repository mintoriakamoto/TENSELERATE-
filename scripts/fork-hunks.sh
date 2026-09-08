#!/usr/bin/env bash
# Fork footprint: where this repository differs from upstream llama.cpp inside
# upstream-owned directories, and every hunk marked `TENSELERATE`. This is the
# checklist for an upstream sync: each file below is a potential conflict, and
# each marked hunk is what to re-apply if the resolution takes upstream's side.
#
# Usage: scripts/fork-hunks.sh [upstream-ref]   (default: upstream/master)
#   git remote add upstream https://github.com/ggml-org/llama.cpp.git && git fetch upstream master
set -euo pipefail
ref="${1:-upstream/master}"
cd "$(dirname "$0")/.."

if ! git rev-parse --verify -q "$ref" >/dev/null; then
    echo "error: $ref not found; git remote add upstream https://github.com/ggml-org/llama.cpp.git && git fetch upstream master" >&2
    exit 1
fi

upstream_dirs=(src ggml common tools include tests examples gguf-py convert_hf_to_gguf.py CMakeLists.txt)

echo "== files that differ from $ref in upstream-owned directories"
git diff --stat "$ref" -- "${upstream_dirs[@]}" | sed '$d' | sort -t'|' -k2 -rn
echo
echo "== fork-only files inside upstream-owned directories (no conflict, but re-check they still build)"
comm -23 <(git ls-tree -r --name-only HEAD -- "${upstream_dirs[@]}" | sort) \
         <(git ls-tree -r --name-only "$ref" -- "${upstream_dirs[@]}" | sort) \
    | grep -v '^tests/tenselerate/' || true
echo
echo "== TENSELERATE-marked hunks (what to re-apply if a conflict is resolved to upstream)"
git grep -n "TENSELERATE" -- "${upstream_dirs[@]}" ':!tests/tenselerate' ':!src/tenselerate-*' \
    | grep -v -i "tenselerate-update\|tenselerate boot\|tenselerate serve" || true
