#!/usr/bin/env bash
# PHANTOM One-Command Installer (Linux, macOS, WSL2)
set -e

echo "============================================================"
echo "  PHANTOM — Universal Model Runtime Platform Installer"
echo "  Run the Unreachable."
echo "============================================================"

# Check for WSL2 if on Windows
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "✓ WSL2 environment detected"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3.10+ is required."
    exit 1
fi

echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -e python/

# Create PHANTOM home
PHANTOM_DIR="$HOME/.phantom"
mkdir -p "$PHANTOM_DIR/models" "$PHANTOM_DIR/downloads" "$PHANTOM_DIR/logs" "$PHANTOM_DIR/rag"

echo "Running system diagnostics..."
python3 -m phantom.phantom_cli doctor

echo ""
echo "============================================================"
echo "✓ PHANTOM is ready!"
echo "Try running:"
echo "    phantom plan llama3:70b"
echo "    phantom serve"
echo "============================================================"
