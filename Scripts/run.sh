#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/Models/embeddinggemma-300m-coreml"

# Download the ANE-optimized CoreML bundle if it isn't there yet.
if [ ! -d "$MODEL_DIR/encoder.mlmodelc" ]; then
    echo "[↓] Bundle not found. Running download script..."
    bash "$SCRIPT_DIR/download-model.sh"
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Starting EmbeddingGemma-300M Server                 ║"
echo "║  Powered by Apple Neural Engine                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

exec swift run EmbeddingServer \
    --port "${PORT:-6333}" \
    --model "$MODEL_DIR/encoder.mlmodelc" \
    --compute-units cpuAndNeuralEngine \
    "$@"
