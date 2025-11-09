#!/bin/bash
# Native installation script for vLLM on ROCm
# This builds vLLM from source for your gfx1100 GPU

set -e

echo "🔧 Building vLLM for ROCm (gfx1100)..."
echo "⚠️  This may take 10-20 minutes"

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for ROCm
echo "Installing PyTorch for ROCm..."
pip uninstall torch -y || true
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Build & install AMD SMI
echo "Installing AMD SMI..."
pip install /opt/rocm/share/amd_smi

# Install Triton for ROCm
echo "Installing Triton for ROCm..."
pip install ninja cmake wheel pybind11
pip uninstall -y triton || true

cd /tmp
if [ -d "triton" ]; then
    rm -rf triton
fi

git clone https://github.com/ROCm/triton.git
cd triton
git checkout f9e5bf54
if [ ! -f setup.py ]; then cd python; fi
python3 setup.py install
cd ../..

# Install flash attention (optional)
echo "Installing Flash Attention for ROCm..."
cd /tmp
if [ -d "flash-attention" ]; then
    rm -rf flash-attention
fi

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 0e60e394
git submodule update --init
GPU_ARCHS="gfx1100" python3 setup.py install
cd ../..

# Return to project directory
cd "$OLDPWD"

# Install dependencies
echo "Installing vLLM dependencies..."
pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools_scm

# Clone vLLM if not exists
if [ ! -d "vllm-source" ]; then
    echo "Cloning vLLM..."
    git clone https://github.com/vllm-project/vllm.git vllm-source
fi

cd vllm-source

# Install ROCm requirements
pip install -r requirements/rocm.txt

# Set architecture for your GPU (gfx1100 = Radeon RX 7900 series)
export PYTORCH_ROCM_ARCH="gfx1100"

# Build vLLM
echo "Building vLLM for gfx1100..."
python3 setup.py develop

cd ..

echo "✅ vLLM installation completed!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation:"
echo "  python examples/simple_inference.py"
