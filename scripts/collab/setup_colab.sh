#!/bin/bash
# scripts/collab/setup_colab.sh
# Setup environment for Calculus of Learning on Google Colab

set -e

echo "========================================================="
echo "🚀 Starting Colab Environment Setup for Calculus of Learning"
echo "========================================================="

# 1. Check GPU Status
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️ WARNING: No GPU detected via nvidia-smi. If you intend to use GPU acceleration (highly recommended), please change the Colab runtime type to T4 GPU, L4 GPU, or A100 GPU."
fi

# 2. Install System Dependencies for Headless Rendering
echo "📦 Installing system dependencies (OSMesa, GLX, FFmpeg)..."
sudo apt-get update -y
sudo apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libosmesa6-dev \
    patchelf \
    freeglut3-dev \
    mesa-utils

# 3. Install uv for fast dependency management
echo "⚡ Installing uv..."
pip install --upgrade uv

# 4. Install repository in editable mode along with base dependencies
echo "🐍 Installing project dependencies with uv..."
uv pip install --system -e .

# 5. Install third_party DreamerV3 requirements
if [ -f "third_party/dreamerv3/requirements.txt" ]; then
    echo "🤖 Installing DreamerV3 requirements..."
    uv pip install --system -r third_party/dreamerv3/requirements.txt
else
    echo "⚠️ third_party/dreamerv3/requirements.txt not found, skipping."
fi

# 6. Reinstall/Upgrade JAX with CUDA 12 support (Colab runs CUDA 12)
echo "🔥 Upgrading JAX with GPU support..."
uv pip install --system -U "jax[cuda12]" flax optax

# 7. Configure environment variables (EGL rendering and UV system Python)
echo "🖥️ Setting up environment variables..."
export MUJOCO_GL="egl"
export UV_SYSTEM_PYTHON=1
echo "MUJOCO_GL set to EGL for headless rendering."
echo "UV_SYSTEM_PYTHON set to 1 to run uv commands on the system Python environment."

# 8. Verification Python code
echo "🧪 Running verification tests..."
uv run python -c "
import sys
print(f'Python Version: {sys.version}')

import jax
print(f'JAX Version: {jax.__version__}')
devices = jax.devices()
print(f'JAX Devices: {devices}')
assert any(d.device_kind.lower() != 'cpu' for d in devices), 'GPU not detected by JAX!'
print('✅ JAX successfully detected GPU.')

import mujoco
print(f'MuJoCo Version: {mujoco.__version__}')
# Test rendering compilation
import numpy as np
try:
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    data = mujoco.MjData(model)
    with mujoco.Renderer(model) as renderer:
        renderer.render()
    print('✅ MuJoCo headless rendering test succeeded.')
except Exception as e:
    print(f'⚠️ MuJoCo rendering verification failed: {e}')
"

echo "========================================================="
echo "🎉 Setup Completed Successfully!"
echo "Please restart the Colab kernel if imports do not resolve."
echo "========================================================="
