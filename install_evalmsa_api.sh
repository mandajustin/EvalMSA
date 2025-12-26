#!/bin/bash

# ============================================================
# Eval_MSA API Automated Installation Script
# Tested on Ubuntu 20.04 / 22.04
# ============================================================

set -e  # Exit immediately if a command fails

echo "=============================================="
echo " Eval_MSA API - Automated Installation Started "
echo "=============================================="

# -------- 1. Check OS --------
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
  echo " This script is intended for Linux systems only."
  exit 1
fi

# -------- 2. Update system --------
echo "🔄 Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# -------- 3. Install core system dependencies --------
echo " Installing system dependencies..."
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  build-essential \
  wget \
  curl

# -------- 4. Install MSA tools --------
echo " Installing Multiple Sequence Alignment (MSA) tools..."

sudo apt install -y \
  mafft \
  clustalo \
  muscle \
  t-coffee \
  probcons \
  kalign

# PRANK is not always available via apt in some repos
if ! command -v prank &> /dev/null; then
  echo "Installing PRANK manually..."
  sudo apt install -y prank || {
    echo "⚠️ PRANK not available via apt. Please install manually if needed."
  }
fi

# -------- 5. Verify MSA tools --------
echo "🔍 Verifying MSA tools installation..."
TOOLS=("mafft" "clustalo" "muscle" "t_coffee" "probcons" "kalign" "prank")

for tool in "${TOOLS[@]}"; do
  if command -v "$tool" &> /dev/null; then
    echo " $tool found"
  else
    echo "⚠️ $tool NOT found in PATH"
  fi
done

# -------- 6. Clone EvalMSA repository --------
INSTALL_DIR="$HOME/EvalMSA"

if [ -d "$INSTALL_DIR" ]; then
  echo " EvalMSA directory already exists. Skipping clone."
else
  echo "⬇️ Cloning EvalMSA repository..."
  git clone https://github.com/mandajustin/EvalMSA.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# -------- 7. Validate repository structure --------
echo " Verifying backend source files..."

REQUIRED_FILES=("main.py" "requirements.txt" "evaluator.py" "email_service.py")

for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo " Missing required file: $file"
    exit 1
  fi
done

echo " All required backend files found"

# -------- 8. Create Python virtual environment --------
if [ ! -d "venv" ]; then
  echo "🐍 Creating Python virtual environment..."
  python3 -m venv venv
else
  echo "🐍 Virtual environment already exists"
fi

# -------- 9. Activate virtual environment --------
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# -------- 10. Upgrade pip --------
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# -------- 11. Install Python dependencies --------
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# -------- 12. Installation summary --------
echo ""
echo "=============================================="
echo " ✅ Eval_MSA API Installation Completed"
echo "=============================================="
echo ""
echo "To start the API server:"
echo ""
echo "  cd ~/EvalMSA"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo ""
echo "Health check endpoint:"
echo "  http://localhost:8000/health"
echo "=============================================="
