#!/usr/bin/env bash
# prune-releases - delete releases that carry none of this fork's binaries.
#
# On 2026-09-09 this repository held 31 releases totalling 58.1 GB, and 45.5 GB
# of that was upstream llama.cpp's entire release matrix, inherited from before
# the fork replaced upstream's release workflow with its own: Windows CUDA,
# ROCm, SYCL, OpenVINO, Vulkan, macOS, Android, xcframework and two 391 MB
# cudart zips, per release, sixteen times over. None of it runs on a Linux box
# with two NVIDIA cards, and none of it is reachable by
# `scripts/tenselerate-update.sh`, which resolves /releases/latest and fetches
# only `bin-ubuntu-x64` or `bin-ubuntu-cuda-12.8-sm80-86-x64`.
#
# The rule here is deliberately not "older than N". It is: **a release is kept
# if and only if it carries the fork's own CUDA asset.** That keeps every build
# this box could ever install and deletes only artifacts of a workflow the fork
# no longer runs, so it needs no judgement about how far back to keep.
#
# Releases with zero assets are also deleted: they are shells left by runs that
# tagged but never published, and one of them (main-b138-f43b644) is the
# "older llama-bench tarball" the bench docs used to point at - a tarball that
# never existed.
#
# Git tags are left alone. They cost nothing, and keeping them preserves the
# commit each release pointed at.
#
# Usage:
#   bash scripts/prune-releases.sh                # dry run: list, delete nothing
#   bash scripts/prune-releases.sh --yes          # actually delete
#   KEEP_EXTRA=main-b11036-4f8f50e bash scripts/prune-releases.sh --yes
#   bash scripts/prune-releases.sh --self-test    # no network; exit 0
#
# Needs `gh` authenticated with repo scope, or GH_TOKEN set.
# Env: REPO (mintoriakamoto/TENSELERATE-) FORK_ASSET KEEP_EXTRA (comma-separated tags)

set -euo pipefail

REPO="${REPO:-mintoriakamoto/TENSELERATE-}"
FORK_ASSET="${FORK_ASSET:-bin-ubuntu-cuda-12.8-sm80-86}"
KEEP_EXTRA="${KEEP_EXTRA:-}"
YES=""

for arg in "$@"; do
    case "$arg" in
        --yes|-y)     YES=1 ;;
        --self-test)  SELF_TEST=1 ;;
        *) echo "prune-releases: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Reads the /releases JSON on stdin, writes one tab-separated row per release:
# action, tag, id, asset count, bytes, reason.
#
# The python is captured with a quoted heredoc and passed via -c. Two reasons:
# `python3 - <<PY` would make the script itself stdin and eat the JSON, and a
# single-quoted shell string cannot contain an apostrophe, which the reasons
# below do.
CLASSIFY_PY=$(cat <<'PY'
import json, os, sys

fork_asset = os.environ["FORK_ASSET"]
keep_extra = {t.strip() for t in os.environ.get("KEEP_EXTRA", "").split(",") if t.strip()}

for r in json.load(sys.stdin):
    assets = r.get("assets", [])
    size   = sum(a["size"] for a in assets)
    tag    = r["tag_name"]

    if tag in keep_extra:
        action, why = "keep", "named in KEEP_EXTRA"
    elif any(fork_asset in a["name"] for a in assets):
        action, why = "keep", "carries this fork's CUDA asset"
    elif not assets:
        action, why = "delete", "empty shell, no assets"
    else:
        action, why = "delete", "no fork asset (upstream matrix)"

    print("\t".join([action, tag, str(r["id"]), str(len(assets)), str(size), why]))
PY
)

classify() {
    FORK_ASSET="$FORK_ASSET" KEEP_EXTRA="$KEEP_EXTRA" python3 -c "$CLASSIFY_PY"
}

if [ -n "${SELF_TEST:-}" ]; then
    out=$(printf '%s' '[
      {"tag_name":"main-b11094-x","id":1,"assets":[
        {"name":"tenselerate-main-b11094-x-bin-ubuntu-cuda-12.8-sm80-86-x64.tar.gz","size":1140000000},
        {"name":"tenselerate-main-b11094-x-bin-ubuntu-x64.tar.gz","size":11000000}]},
      {"tag_name":"main-b102-y","id":2,"assets":[
        {"name":"llama-main-b102-y-bin-win-cuda-12.4-x64.zip","size":276000000},
        {"name":"cudart-llama-bin-win-cuda-13.3-x64.zip","size":391000000}]},
      {"tag_name":"main-b138-z","id":3,"assets":[]},
      {"tag_name":"protected-tag","id":4,"assets":[
        {"name":"llama-main-bin-win-sycl-x64.zip","size":115000000}]}
    ]' | KEEP_EXTRA=protected-tag classify)

    got=$(printf '%s\n' "$out" | awk '{print $1"/"$2}' | tr '\n' ' ')
    want="keep/main-b11094-x delete/main-b102-y delete/main-b138-z keep/protected-tag "
    [ "$got" = "$want" ] || { echo "self-test FAILED: $got" >&2; exit 1; }

    # a release whose only asset is upstream's must not be kept by a substring
    # accident: "bin-ubuntu-x64" appears inside the fork's name too
    one=$(printf '%s' '[{"tag_name":"t","id":9,"assets":[{"name":"llama-t-bin-ubuntu-x64.tar.gz","size":1}]}]' | classify)
    case "$one" in delete*) ;; *) echo "self-test FAILED: CPU-only release should be deleted" >&2; exit 1 ;; esac

    echo "prune-releases: self-test OK (keep/delete split, KEEP_EXTRA, no substring false-keep)"
    exit 0
fi

command -v gh >/dev/null || { echo "prune-releases: gh is required" >&2; exit 1; }

rows=$(gh api "repos/$REPO/releases?per_page=100" --paginate | classify)

keep_n=$(printf '%s\n' "$rows" | grep -c '^keep' || true)
del_n=$(printf  '%s\n' "$rows" | grep -c '^delete' || true)
del_b=$(printf  '%s\n' "$rows" | awk -F'\t' '$1=="delete"{s+=$5} END{printf "%.1f", s/1e9}')

printf '\nKEEP (%s):\n' "$keep_n"
printf '%s\n' "$rows" | awk -F'\t' '$1=="keep"{printf "   %-26s %2s assets  %5.2f GB  %s\n", $2, $4, $5/1e9, $6}'
printf '\nDELETE (%s, reclaims %s GB):\n' "$del_n" "$del_b"
printf '%s\n' "$rows" | awk -F'\t' '$1=="delete"{printf "   %-26s %2s assets  %5.2f GB  %s\n", $2, $4, $5/1e9, $6}'

if [ "$keep_n" -eq 0 ]; then
    echo "
prune-releases: REFUSING - nothing matched the keep rule. That means the asset
name changed (FORK_ASSET=$FORK_ASSET) and this would delete everything." >&2
    exit 1
fi

if [ -z "$YES" ]; then
    echo "
Dry run. Re-run with --yes to delete. Tags are left in place either way."
    exit 0
fi

fail=0
while IFS=$'\t' read -r action tag id _ _ _; do
    [ "$action" = "delete" ] || continue
    if gh api -X DELETE "repos/$REPO/releases/$id" >/dev/null 2>&1; then
        echo "   deleted $tag"
    else
        echo "   FAILED  $tag" >&2
        fail=$((fail + 1))
    fi
done <<< "$rows"

[ "$fail" -eq 0 ] || { echo "prune-releases: $fail deletion(s) failed" >&2; exit 1; }
echo "prune-releases: done, ~$del_b GB reclaimed"
