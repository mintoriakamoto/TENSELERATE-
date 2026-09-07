#!/usr/bin/env bash
# Package a release tarball the way scripts/tenselerate-update.sh expects it:
#   tenselerate-<tag>-bin-ubuntu-x64.tar.gz                (flavor "cpu")
#   tenselerate-<tag>-bin-ubuntu-<flavor>-x64.tar.gz       (any other flavor)
# Contents: bin/ (llama-server, llama-cli), lib/ (kernel bridge .so when built),
# the tenselerate Python package, the lifecycle scripts, and the operator docs -
# so an unpacked download runs `python3 -m tenselerate` with only numpy.
# Usage: package-release.sh <tag> <flavor>      (run from the repo root)
set -euo pipefail
TAG="${1:?tag}"; FLAVOR="${2:?flavor}"
if [ "$FLAVOR" = "cpu" ]; then NAME="tenselerate-${TAG}-bin-ubuntu-x64"
else NAME="tenselerate-${TAG}-bin-ubuntu-${FLAVOR}-x64"; fi
STAGE="dist/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE/bin" "$STAGE/lib" "$STAGE/scripts"
for b in llama-server llama-cli; do
    test -x "build/bin/$b" || { echo "missing build/bin/$b" >&2; exit 1; }
    cp "build/bin/$b" "$STAGE/bin/"
done
if [ -f build-kernels/libtenselerate_int8_gemm_c.so ]; then
    cp build-kernels/libtenselerate_int8_gemm_c.so "$STAGE/lib/"
fi
cp -r tenselerate "$STAGE/tenselerate"
find "$STAGE/tenselerate" -name '__pycache__' -type d -prune -exec rm -rf {} +
cp scripts/tenselerate-update.sh scripts/tenselerate-build.sh scripts/hercules_serve.sh scripts/hercules_side_serve.sh "$STAGE/scripts/"
cp HERCULES.md "$STAGE/"
mkdir -p "$STAGE/docs" "$STAGE/benches"
cp docs/rig-cmp170hx-3060.md docs/tenselerate-engine.md "$STAGE/docs/"
cp -r benches/cmp170hx-3060 "$STAGE/benches/"
cp LICENSE "$STAGE/" 2>/dev/null || true
git rev-parse HEAD > "$STAGE/COMMIT"
echo "$TAG" > "$STAGE/VERSION"
tar -C dist -czf "dist/$NAME.tar.gz" "$NAME"
rm -rf "$STAGE"
ls -la dist
