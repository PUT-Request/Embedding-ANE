#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/Models"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Downloading EmbeddingGemma-300M CoreML (ANE)        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

MODEL_REPO="valindotai/embeddinggemma-300m-coreml"
MODEL_DIR="$MODELS_DIR/embeddinggemma-300m-coreml"

if [ -d "$MODEL_DIR/encoder.mlmodelc" ]; then
    echo "[✓] Bundle already exists at: $MODEL_DIR"
    echo "    Remove it first to re-download: rm -rf $MODEL_DIR"
    exit 0
fi

if ! command -v python3 &> /dev/null; then
    echo "[✗] python3 not found. Install it first."
    exit 1
fi

echo "[↓] Downloading $MODEL_REPO via huggingface_hub..."
python3 - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_REPO", local_dir="$MODEL_DIR")
print("downloaded to:", p)
PY

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓ Bundle downloaded successfully!                   ║"
echo "║  Location: $MODEL_DIR/encoder.mlmodelc"
echo "╚══════════════════════════════════════════════════════╝"
