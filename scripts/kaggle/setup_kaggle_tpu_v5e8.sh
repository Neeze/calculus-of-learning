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
    echo "🤖 Installing DreamerV3 requirements (excluding its jax pin)..."
    # DreamerV3 pins `jax[cuda12]==0.4.33`. On TPU that pin is actively harmful:
    # it drags in jax-cuda12-* 0.4.33 and leaves a stale libtpu 0.0.44 behind,
    # which then mismatches the jaxlib we install below and SEGFAULTS at the
    # first jax.devices() call. Strip the jax line and let step 5 own JAX.
    grep -v -E '^\s*jax(\[|=|>|<|\s|$)' third_party/dreamerv3/requirements.txt \
        > /tmp/dreamerv3-reqs-nojax.txt
    uv pip install --system -r /tmp/dreamerv3-reqs-nojax.txt
else
    echo "⚠️ third_party/dreamerv3 still missing after init attempt — E1 Tier 1 (DreamerV3) will be unavailable."
fi

# 5. JAX for TPU. Install unconditionally and as the LAST step, so jax, jaxlib
# and libtpu all come from one resolution and cannot drift apart. Any CUDA jax
# plugin left over from the base image is removed first — on TPU it is unused,
# and a version-mismatched one both spams warnings and risks loader conflicts.
echo "🧹 Removing CUDA JAX plugins and any stale libtpu..."
pip uninstall -y -q jax-cuda12-plugin jax-cuda12-pjrt libtpu 2>/dev/null || true

echo "🔥 Installing jax[tpu] (pulls a matching libtpu)..."
uv pip install --system -U "jax[tpu]"

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
