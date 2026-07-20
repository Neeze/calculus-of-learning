#!/bin/bash
# scripts/kaggle/setup_kaggle_rtx6000.sh
# Setup environment for Calculus of Learning on Kaggle GPU RTX 6000 (single,
# large-VRAM workstation GPU — 48GB class).
#
# Prereqs (Kaggle notebook settings, right sidebar):
#   - Accelerator: GPU RTX 6000 (or whichever single large-VRAM GPU is listed
#     under your Kaggle account — this script is CUDA-generic, not tied to
#     one specific SKU name)
#   - Internet: ON (required for pip/git — off by default on Kaggle)

set -e

echo "========================================================="
echo "🚀 Kaggle GPU RTX 6000 Environment Setup for Calculus of Learning"
echo "========================================================="

SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null && SUDO="sudo"

# 1. Check GPU status
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "⚠️ WARNING: No GPU detected via nvidia-smi. Check the accelerator"
    echo "   setting in the notebook sidebar."
fi

# 2. System dependencies for headless rendering
echo "📦 Installing system dependencies (EGL/OSMesa, FFmpeg)..."
$SUDO apt-get update -y -q
$SUDO apt-get install -y -q ffmpeg libosmesa6-dev patchelf libegl1 libgl1 > /dev/null \
    || $SUDO apt-get install -y -q ffmpeg libosmesa6-dev patchelf libgl1-mesa-glx > /dev/null

# 3. uv for fast dependency management
echo "⚡ Installing uv..."
pip install --upgrade -q uv
export UV_SYSTEM_PYTHON=1

# 4. Project dependencies
echo "🐍 Installing project dependencies..."
uv pip install --system -e .

# 5. DreamerV3 requirements (E1 Tier 1)
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

# 6. JAX with CUDA 12 support — must be last install step, since dreamerv3's
# requirements can otherwise downgrade/replace it.
echo "🔥 Installing JAX with CUDA 12 support..."
uv pip install --system -U "jax[cuda12]" flax optax

# 7. Environment variables
export MUJOCO_GL="egl"
echo "MUJOCO_GL=egl (headless rendering on GPU)"

# 8. Verification
echo "🧪 Verifying..."
python -c "
import jax
print(f'JAX Version: {jax.__version__}')
devices = jax.devices()
print(f'JAX Devices: {devices}')
assert any(d.platform == 'gpu' for d in devices), 'No GPU detected by JAX!'
print('✅ JAX successfully detected GPU.')
for d in devices:
    if d.platform == 'gpu':
        stats = d.memory_stats() or {}
        limit = stats.get('bytes_limit')
        if limit:
            print(f'   {d}: ~{limit / 1e9:.1f} GB visible to JAX')

from dm_control import suite
env = suite.load('cartpole', 'swingup')
env.reset()
print('✅ dm_control loads and steps.')

import mujoco
try:
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    with mujoco.Renderer(model) as renderer:
        renderer.render()
    print('✅ MuJoCo headless rendering (EGL) works.')
except Exception as e:
    print(f'⚠️ Rendering check failed (state-based experiments unaffected): {e}')
"

echo "========================================================="
echo "🎉 Kaggle GPU RTX 6000 Setup Completed!"
echo "NOTE: set this in every notebook cell that runs experiments:"
echo "  %env MUJOCO_GL=egl"
echo ""
echo "Single large-VRAM GPU: no parallel-seed sharding needed — a run's"
echo "batch_size can be raised (e.g. --batch_size 32) to use the extra"
echo "headroom instead. See scripts/kaggle/README.md."
echo "========================================================="
