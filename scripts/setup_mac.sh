#!/usr/bin/env bash
# setup_mac.sh – One-shot setup for grid-cv on Apple Silicon (M1/M2/M3/M4)
#
# Usage:
#   chmod +x scripts/setup_mac.sh
#   ./scripts/setup_mac.sh
#
# What it does:
#   1. Checks for Homebrew, installs if missing
#   2. Installs Python 3.11 via Homebrew (stable, well-tested with PyTorch MPS)
#   3. Creates a .venv virtualenv in the repo root
#   4. Installs all dependencies including PyTorch with MPS support
#   5. Verifies MPS is available
#   6. Runs pytest to confirm everything works

set -euo pipefail

PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "================================================"
echo "  grid-cv  Mac Apple Silicon Setup"
echo "  Repo: $REPO_ROOT"
echo "================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Homebrew
# ---------------------------------------------------------------------------
if ! command -v brew &>/dev/null; then
  echo "[1/6] Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for the rest of this script
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "[1/6] Homebrew already installed: $(brew --version | head -1)"
fi

# ---------------------------------------------------------------------------
# 2. Python
# ---------------------------------------------------------------------------
if ! brew list python@${PYTHON_VERSION} &>/dev/null; then
  echo "[2/6] Installing Python ${PYTHON_VERSION}..."
  brew install python@${PYTHON_VERSION}
else
  echo "[2/6] Python ${PYTHON_VERSION} already installed"
fi

PYTHON="$(brew --prefix)/bin/python${PYTHON_VERSION}"

# ---------------------------------------------------------------------------
# 3. Virtual environment
# ---------------------------------------------------------------------------
VENV="$REPO_ROOT/.venv"
if [ ! -d "$VENV" ]; then
  echo "[3/6] Creating virtualenv at .venv..."
  "$PYTHON" -m venv "$VENV"
else
  echo "[3/6] Virtualenv already exists at .venv"
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ---------------------------------------------------------------------------
# 4. Install dependencies
# ---------------------------------------------------------------------------
echo "[4/6] Installing dependencies (this may take a few minutes)..."
"$PIP" install --upgrade pip --quiet

# PyTorch — MPS support ships in the standard macOS wheel since 2.0
"$PIP" install torch torchvision --quiet

# Main project + dev extras
"$PIP" install -e "$REPO_ROOT[dev]" --quiet

# Training extras
"$PIP" install \
  ultralytics \
  sahi \
  albumentations \
  huggingface_hub \
  gdown \
  pyyaml \
  pandas \
  tqdm \
  --quiet

echo "    Dependencies installed."

# ---------------------------------------------------------------------------
# 5. Verify MPS
# ---------------------------------------------------------------------------
echo "[5/6] Verifying MPS (Metal Performance Shaders) availability..."
"$PY" - <<'EOF'
import torch
if torch.backends.mps.is_available():
    print("    MPS is AVAILABLE - GPU training enabled")
    x = torch.ones(3, device="mps")
    print(f"    Test tensor on MPS: {x}")
else:
    print("    WARNING: MPS not available - will fall back to CPU")
    print("    Make sure you are on macOS 12.3+ and Python is not Rosetta")
EOF

# ---------------------------------------------------------------------------
# 6. Run tests
# ---------------------------------------------------------------------------
echo "[6/6] Running pytest..."
"$VENV/bin/pytest" "$REPO_ROOT/tests" -q

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Download data (free, no API keys, ~7 GB):"
echo "    python scripts/download_insplad.py"
echo "    python scripts/download_mpid.py"
echo "    python scripts/build_training_set.py --name poc"
echo ""
echo "  Then train:"
echo "    python scripts/train.py detector --data datasets/build/poc/detector.yaml"
echo "    python scripts/train.py classifier --data datasets/build/poc/classifier"
echo "================================================"
echo ""
