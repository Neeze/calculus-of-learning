#!/bin/bash
# scripts/kaggle/setup_kaggle_tpu_v5e8.sh
# Setup environment for Calculus of Learning on Kaggle TPU v5e-8 (8 chips).
#
# Prereqs (Kaggle notebook settings, right sidebar):
#   - Accelerator: TPU v5e-8
#   - Internet: ON (required for pip/git — off by default on Kaggle)

set -e

echo "========================================================="
echo "🚀 Kaggle TPU v5e-8 Environment Setup for Calculus of Learning"
echo "========================================================="

# Kaggle kernels already run as root — sudo may not even exist.
SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null && SUDO="sudo"

# 1. System dependencies (headless MuJoCo — TPU host has no GPU, use OSMesa)
echo "📦 Installing system dependencies..."
$SUDO apt-get update -y -q
$SUDO apt-get install -y -q ffmpeg libosmesa6-dev patchelf > /dev/null

# 2. uv for fast dependency management
echo "⚡ Installing uv..."
pip install --upgrade -q uv
export UV_SYSTEM_PYTHON=1

# 3. Project dependencies
echo "🐍 Installing project dependencies..."
uv pip install --system -e .

# 4. DreamerV3 requirements (E1 Tier 1)
echo "🔗 Initializing third_party/dreamerv3 submodule..."
if [ ! -f "third_party/dreamerv3/requirements.txt" ]; then
    if [ -d ".git" ] && [ -f ".gitmodules" ]; then
        git submodule update --init --recursive third_party/dreamerv3
    else
        echo "Not a git checkout with .gitmodules — cloning directly instead."
        rm -rf third_party/dreamerv3
        git clone https://github.com/danijar/dreamerv3.git third_party/dreamerv3
    fi
fi

if [ -f "third_party/dreamerv3/requirements.txt" ]; then
    echo "🤖 Installing DreamerV3 requirements..."
    uv pip install --system -r third_party/dreamerv3/requirements.txt
else
    echo "⚠️ third_party/dreamerv3 still missing after init attempt — E1 Tier 1 (DreamerV3) will be unavailable."
fi

# 5. JAX for TPU. Kaggle's TPU VM image usually ships a working jax+libtpu
# already wired to the TPU runtime; only force-reinstall if the pre-shipped
# one doesn't see the TPU, since reinstalling can desync from the host's
# libtpu version and break it.
echo "🔥 Checking JAX/TPU..."
if python -c "import jax; assert any('tpu' in d.platform.lower() for d in jax.devices())" 2>/dev/null; then
    echo "✅ Pre-installed JAX already sees the TPU — leaving it as is."
else
    echo "Pre-installed JAX does not see the TPU, installing jax[tpu]..."
    uv pip install --system -U "jax[tpu]" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

# 6. Environment variables
export MUJOCO_GL="osmesa"
export JAX_PLATFORMS="tpu"
echo "MUJOCO_GL=osmesa (headless, no GPU on TPU host)"
echo "JAX_PLATFORMS=tpu"

# 7. Verification
echo "🧪 Verifying..."
python -c "
import jax
print(f'JAX Version: {jax.__version__}')
devices = jax.devices()
print(f'JAX Devices ({len(devices)}): {devices}')
assert any('tpu' in d.platform.lower() for d in devices), 'TPU not detected by JAX!'
print(f'✅ JAX detected {len(devices)} TPU chip(s).')

from dm_control import suite
env = suite.load('cartpole', 'swingup')
env.reset()
print('✅ dm_control loads and steps without rendering.')
"

echo "========================================================="
echo "🎉 Kaggle TPU v5e-8 Setup Completed!"
echo "NOTE: set these in every notebook cell that runs experiments:"
echo "  %env MUJOCO_GL=osmesa"
echo "  %env JAX_PLATFORMS=tpu"
echo ""
echo "v5e-8 gives 8 TPU chips visible as jax.devices() — see"
echo "scripts/kaggle/README.md for how to run E1/E2 seeds in parallel."
echo "========================================================="
