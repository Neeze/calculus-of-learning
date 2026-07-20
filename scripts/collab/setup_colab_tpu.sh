#!/bin/bash
# scripts/collab/setup_colab_tpu.sh
# Setup environment for Calculus of Learning on Google Colab TPU (v5e-1 / v2-8)

set -e

echo "========================================================="
echo "🚀 Colab TPU Environment Setup for Calculus of Learning"
echo "========================================================="

# 1. System dependencies (headless MuJoCo — TPU runtime has no GPU, use OSMesa)
echo "📦 Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y ffmpeg libosmesa6-dev patchelf > /dev/null

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
    # it drags in jax-cuda12-* 0.4.33 and leaves a stale libtpu behind, which
    # then mismatches the jaxlib installed below and SEGFAULTS at the first
    # jax.devices() call. Strip the jax line and let step 5 own JAX.
    grep -v -E '^\s*jax(\[|=|>|<|\s|$)' third_party/dreamerv3/requirements.txt \
        > /tmp/dreamerv3-reqs-nojax.txt
    uv pip install --system -r /tmp/dreamerv3-reqs-nojax.txt
else
    echo "⚠️ third_party/dreamerv3 still missing after init attempt — E1 Tier 1 (DreamerV3) will be unavailable."
fi

# 5. Replace CUDA jax (pulled by pyproject) with TPU jax — MUST be last install
# step, so jax, jaxlib and libtpu all come from a single resolution and cannot
# drift apart (a stale libtpu segfaults at jax.devices()).
echo "🔥 Installing JAX for TPU..."
uv pip uninstall --system jax-cuda12-plugin jax-cuda12-pjrt libtpu 2>/dev/null || true
uv pip install --system -U "jax[tpu]"

# 6. Environment variables
export MUJOCO_GL="osmesa"
export JAX_PLATFORMS="tpu"
echo "MUJOCO_GL=osmesa (headless, no GPU on TPU runtime)"
echo "JAX_PLATFORMS=tpu"

# 7. Verification
echo "🧪 Verifying..."
python -c "
import jax
print(f'JAX Version: {jax.__version__}')
devices = jax.devices()
print(f'JAX Devices: {devices}')
assert any('tpu' in d.platform.lower() for d in devices), 'TPU not detected by JAX!'
print('✅ JAX successfully detected TPU.')

from dm_control import suite
env = suite.load('cartpole', 'swingup')
env.reset()
print('✅ dm_control loads and steps without rendering.')
"

echo "========================================================="
echo "🎉 TPU Setup Completed!"
echo "NOTE: set these in every notebook cell that runs experiments:"
echo "  %env MUJOCO_GL=osmesa"
echo "  %env JAX_PLATFORMS=tpu"
echo "========================================================="
