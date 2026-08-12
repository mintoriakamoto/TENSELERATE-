#!/usr/bin/env bash
# tenselerate-update - check for and apply updates published by this fork.
#
# Every native-code push to main auto-publishes a `main-b<N>-<sha>` release with
# binaries for each platform (.github/workflows/release.yml). This is the pull
# side of that: it compares what you are running against the newest release and
# either fast-forwards + rebuilds the source tree, or downloads the prebuilt
# binary for this machine.
#
#   scripts/tenselerate-update.sh --check     # what is available? (exit 10 = update)
#   scripts/tenselerate-update.sh --source    # fast-forward main and rebuild
#   scripts/tenselerate-update.sh --binary    # download the release build instead
#   scripts/tenselerate-update.sh --list      # show the release's assets
#
# Works without a git clone: downloaders can curl this script on its own, and
# the local version is then read from `llama-cli --version` ($LLAMA_BIN,
# build/bin, an unpacked dist/, or PATH).
#
# Override the upstream with TENSELERATE_REPO=owner/name (defaults to this
# clone's origin, then to the fork).

set -euo pipefail

DEFAULT_REPO="mintoriakamoto/TENSELERATE-"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
DEST="${DEST:-$ROOT/dist}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

repo_slug() {
    if [ -n "${TENSELERATE_REPO:-}" ]; then
        printf '%s\n' "$TENSELERATE_REPO"
        return
    fi
    url=$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)
    case "$url" in
        *github.com[:/]*)
            printf '%s\n' "$url" | sed -E 's#.*github\.com[:/]([^/]+/[^/]+?)(\.git)?/?$#\1#'
            ;;
        *) printf '%s\n' "$DEFAULT_REPO" ;;
    esac
}

api() {
    have curl || die "curl is required"
    curl -fsSL ${GITHUB_TOKEN:+-H "Authorization: Bearer $GITHUB_TOKEN"} \
        -H "Accept: application/vnd.github+json" "$1"
}

# newest published release, "tag<TAB>published_at"
latest_release() {
    api "https://api.github.com/repos/$1/releases/latest" |
        tr ',' '\n' |
        sed -n -e 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/tag \1/p' \
               -e 's/.*"published_at"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/date \1/p'
}

release_assets() {
    api "https://api.github.com/repos/$1/releases/latest" |
        tr ',' '\n' |
        sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

# build number out of a main-b<N>-<sha> tag; empty when the tag is not that shape
tag_build_number() { printf '%s\n' "$1" | sed -n 's/^main-b\([0-9]\+\)-.*/\1/p'; }

local_sha() { git -C "$ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo unknown; }

# build number of the source tree; CI stamps the commit count on main
source_build_number() { git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo ""; }

# the binary actually installed here - a release download has no git clone, so
# this is the only version it can report. `--version` prints "version: N (sha)".
find_binary() {
    for cand in "${LLAMA_BIN:-}" "$BUILD_DIR/bin/llama-cli" "$ROOT/build/bin/llama-cli" \
                "$DEST"/*/llama-cli "$ROOT"/llama-cli; do
        [ -n "$cand" ] && [ -x "$cand" ] && { printf '%s\n' "$cand"; return 0; }
    done
    command -v llama-cli 2>/dev/null || true
}

# "<build> <sha>" of the installed binary, empty when there is none
binary_version() {
    bin=$(find_binary)
    [ -n "$bin" ] || return 0
    "$bin" --version 2>&1 | sed -n 's/^version: \([0-9]\+\) (\([^)]*\)).*/\1 \2/p' | head -n 1
}

# how this machine appears in release asset names: "<os> <arch>"
platform_tokens() {
    os=$(uname -s); arch=$(uname -m)
    case "$os" in
        Linux)  os_tok="ubuntu" ;;
        Darwin) os_tok="macos" ;;
        MINGW*|MSYS*|CYGWIN*) os_tok="win" ;;
        *) die "unsupported OS for --binary: $os (use --source)" ;;
    esac
    case "$arch" in
        x86_64|amd64) arch_tok="x64" ;;
        arm64|aarch64) arch_tok="arm64" ;;
        *) die "unsupported arch for --binary: $arch (use --source)" ;;
    esac
    printf '%s %s\n' "$os_tok" "$arch_tok"
}

cmd_check() {
    slug=$(repo_slug)
    info=$(latest_release "$slug") || die "could not reach the release API for $slug"
    tag=$(printf '%s\n' "$info" | sed -n 's/^tag //p')
    date=$(printf '%s\n' "$info" | sed -n 's/^date //p')
    [ -n "$tag" ] || die "no published release found for $slug"

    printf 'repo    : %s\n' "$slug"

    src_n=$(source_build_number)
    read -r bin_n bin_sha <<<"$(binary_version)"
    if [ -n "${bin_n:-}" ]; then
        printf 'running : b%s (%s)\n' "$bin_n" "${bin_sha%%-*}"
    fi
    if [ -n "$src_n" ]; then
        printf 'source  : b%s (%s)\n' "$src_n" "$(local_sha)"
    fi
    [ -n "${bin_n:-}$src_n" ] || printf 'running : unknown - no binary found and no git clone here\n'
    printf 'latest  : %s  published %s\n' "$tag" "${date:-?}"

    # what you are actually running wins; fall back to the checkout
    local_n=${bin_n:-$src_n}
    remote_n=$(tag_build_number "$tag")
    if [ -z "$remote_n" ] || [ -z "$local_n" ]; then
        printf 'status  : cannot compare automatically - check the release page\n'
        return 0
    fi
    if [ -n "${bin_n:-}" ] && [ -n "$src_n" ] && [ "$src_n" -gt "$bin_n" ]; then
        printf 'note    : the checkout is %d commits ahead of the installed binary - rebuild\n' \
            "$((src_n - bin_n))"
    fi
    if [ "$remote_n" -gt "$local_n" ]; then
        printf 'status  : UPDATE AVAILABLE (%d commits ahead)\n' "$((remote_n - local_n))"
        printf '          apply: %s --source   (or --binary)\n' "$0"
        return 10
    fi
    printf 'status  : up to date\n'
}

cmd_list() {
    slug=$(repo_slug)
    release_assets "$slug" | sed 's#.*/##'
}

cmd_source() {
    git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet ||
        die "working tree is dirty - commit or stash first"

    branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
    [ "$branch" = "main" ] || printf 'note: on branch %s, updating it from origin/main\n' "$branch"

    printf '==> fetching\n'
    git -C "$ROOT" fetch origin main --tags
    printf '==> fast-forwarding\n'
    git -C "$ROOT" merge --ff-only origin/main

    if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
        printf '==> rebuilding %s with its existing configuration\n' "$BUILD_DIR"
        cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    else
        printf 'no build dir at %s - configure one, e.g.:\n' "$BUILD_DIR"
        printf '  cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_DISABLE_DP4A=ON && cmake --build build -j\n'
    fi
    printf '==> now at %s\n' "$(local_sha)"
}

cmd_binary() {
    slug=$(repo_slug)
    read -r os_tok arch_tok <<<"$(platform_tokens)"
    all=$(release_assets "$slug" | grep -E "bin-$os_tok-.*$arch_tok\.(tar\.gz|tgz|zip)$" || true)
    [ -n "$all" ] || die "no $os_tok/$arch_tok asset in the latest release; try --list"

    if [ -n "${FLAVOR:-}" ]; then
        url=$(printf '%s\n' "$all" | grep -E "$FLAVOR" | head -n 1 || true)
        [ -n "$url" ] || die "no asset matching flavor '$FLAVOR'; candidates:
$(printf '%s\n' "$all" | sed 's#.*/#  #')"
    else
        # plain build is bin-<os>-<arch>.<ext>; anything longer is an accelerator variant
        url=$(printf '%s\n' "$all" | grep -E "bin-$os_tok-$arch_tok\.(tar\.gz|tgz|zip)$" | head -n 1 || true)
        if [ -z "$url" ]; then
            printf 'several builds match %s/%s - pick one with FLAVOR=<substring>:\n' \
                "$os_tok" "$arch_tok" >&2
            printf '%s\n' "$all" | sed 's#.*/#  #' >&2
            die "e.g. FLAVOR=cuda-12.4 $0 --binary"
        fi
    fi

    mkdir -p "$DEST"
    file="$DEST/$(basename "$url")"
    printf '==> downloading %s\n' "$(basename "$url")"
    curl -fL --progress-bar -o "$file" "$url"
    printf '==> unpacking into %s\n' "$DEST"
    case "$file" in
        *.tar.gz|*.tgz) tar -xzf "$file" -C "$DEST" ;;
        *.zip) have unzip || die "unzip is required for this asset"; unzip -oq "$file" -d "$DEST" ;;
        *) die "unknown asset format: $file" ;;
    esac
    printf '==> binaries in %s\n' "$DEST"
}

cmd_self_test() {
    [ "$(tag_build_number main-b1234-abc1234)" = "1234" ] || die "tag parse failed"
    [ -z "$(tag_build_number v1.2.3)" ] || die "non-matching tag should not parse"
    out=$(TENSELERATE_REPO=owner/name repo_slug); [ "$out" = "owner/name" ] || die "env override failed"

    # a release download has no clone: the version must come off the binary
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    printf '#!/bin/sh\necho "version: 42 (cafe123)" >&2\n' > "$tmp/llama-cli"
    chmod +x "$tmp/llama-cli"
    read -r n sha <<<"$(LLAMA_BIN=$tmp/llama-cli binary_version)"
    [ "$n" = "42" ] && [ "$sha" = "cafe123" ] || die "binary version parse failed: '$n' '$sha'"
    empty=$(ROOT=$tmp/empty BUILD_DIR=$tmp/empty DEST=$tmp/empty PATH=/nonexistent binary_version)
    [ -z "$empty" ] || die "no binary anywhere should report nothing, got '$empty'"

    printf 'self-test OK: tag parsing, non-matching tag, repo override,\n'
    printf '              binary --version parsing, missing-binary fallback\n'
}

case "${1:---check}" in
    --check)   cmd_check ;;
    --list)    cmd_list ;;
    --source)  cmd_source ;;
    --binary)  cmd_binary ;;
    --self-test) cmd_self_test ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown option: $1 (try --help)" ;;
esac
