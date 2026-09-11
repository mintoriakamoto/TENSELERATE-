#!/bin/bash
# Download the exact GGUF this fork is built and measured against.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_NAME="Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
GGUF_HF_REPO="DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
OUT="${1:-$ROOT/models}"
mkdir -p "$OUT"
if [ -f "$OUT/$GGUF_NAME" ]; then
  echo "already have $OUT/$GGUF_NAME"
  ls -lh "$OUT/$GGUF_NAME"
  exit 0
fi
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$GGUF_HF_REPO" "$GGUF_NAME" --local-dir "$OUT"
elif command -v hf >/dev/null 2>&1; then
  hf download "$GGUF_HF_REPO" "$GGUF_NAME" --local-dir "$OUT"
else
  echo "Install huggingface_hub: pip install -U huggingface_hub"
  echo "Then: huggingface-cli download $GGUF_HF_REPO $GGUF_NAME --local-dir $OUT"
  exit 1
fi
ls -lh "$OUT/$GGUF_NAME"
