#!/usr/bin/env bash
# PHANTOM One-Command Installer (Linux, macOS, WSL2)
set -e

echo "============================================================"
echo "  PHANTOM — Universal Model Runtime Platform Installer"
echo "  Run the Unreachable."
echo "============================================================"

# Check for WSL2 if on Windows
IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "✓ WSL2 environment detected"
    IS_WSL=true
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3.10+ is required."
    exit 1
fi

# Check GPU & Compute Capability
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "✓ NVIDIA GPU Detected: $GPU_NAME"
else
    echo "⚠ No NVIDIA GPU detected via nvidia-smi. Using CPU / Simulated GPU fallback."
fi

# Check Disk Space (Warn if < 50GB)
AVAILABLE_GB=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 50 ]; then
    echo "⚠ Warning: Only ${AVAILABLE_GB}GB free in $HOME. At least 50GB recommended for 70B NVMe swap."
else
    echo "✓ NVMe / Disk Space: ${AVAILABLE_GB}GB available"
fi

echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -e python/

# Create PHANTOM home
PHANTOM_DIR="$HOME/.phantom"
mkdir -p "$PHANTOM_DIR/models" "$PHANTOM_DIR/downloads" "$PHANTOM_DIR/logs" "$PHANTOM_DIR/rag"

# Register systemd user service on Linux (skip on macOS / WSL2)
if [ "$IS_WSL" = false ] && [ "$(uname)" = "Linux" ] && command -v systemctl &> /dev/null; then
    SERVICE_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SERVICE_DIR"
    cat <<EOF > "$SERVICE_DIR/phantom.service"
[Unit]
Description=PHANTOM Model Runtime Platform
After=network.target

[Service]
Type=simple
ExecStart=$(command -v python3) -m phantom.phantom_cli serve --port 11411
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF
    echo "✓ Registered systemd user service: ~/.config/systemd/user/phantom.service"
    echo "  To enable at login: systemctl --user enable --now phantom"
fi

echo "Running system diagnostics..."
python3 -m phantom.phantom_cli doctor

echo ""
echo "============================================================"
echo "✓ PHANTOM is ready!"
echo "Try running:"
echo "    phantom plan llama3:70b"
echo "    phantom serve"
echo "============================================================"
