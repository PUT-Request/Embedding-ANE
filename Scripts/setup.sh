#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  EmbeddingANE Build Setup                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check for Swift
if ! command -v swift &> /dev/null; then
    echo "[✗] Swift not found. Please install Xcode or Swift toolchain."
    echo "    xcode-select --install"
    exit 1
fi

echo "[✓] Swift version: $(swift --version | head -1)"

# Check for git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "[!] Installing git-lfs..."
    if command -v brew &> /dev/null; then
        brew install git-lfs
    fi
fi

echo ""
echo "[1/3] Resolving Swift package dependencies..."
cd "$PROJECT_DIR"
swift package resolve

echo ""
echo "[2/3] Building (release mode)..."
swift build -c release

echo ""
echo "[3/3] Build complete!"

BIN_PATH=$(swift build -c release --show-bin-path)/EmbeddingServer

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓ Build successful!                                ║"
echo "║                                                      ║"
echo "║  Binary: $BIN_PATH"
echo "║                                                      ║"
echo "║  Run:   ./Scripts/run.sh                             ║"
echo "║  Or:    swift run EmbeddingServer --port 6333        ║"
echo "╚══════════════════════════════════════════════════════╝"
