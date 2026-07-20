#!/bin/bash
# scripts/kaggle/setup_kaggle_2xt4.sh
# Setup environment for Calculus of Learning on Kaggle GPU T4 x2.
#
# Prereqs (Kaggle notebook settings, right sidebar):
#   - Accelerator: GPU T4 x2
#   - Internet: ON (required for pip/git — off by default on Kaggle)

set -e

echo "========================================================="
echo "🚀 Kaggle GPU T4 x2 Environment Setup for Calculus of Learning"
echo "========================================================="

SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null && SUDO="sudo"

# 1. Check GPU status
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "⚠️ WARNING: No GPU detected via nvidia-smi. Set Accelerator to 'GPU T4 x2'"
    echo "   in the notebook settings (right sidebar)."
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
print(f'JAX Devices ({len(devices)}): {devices}')
gpu_count = sum(1 for d in devices if d.platform == 'gpu')
assert gpu_count > 0, 'No GPU detected by JAX!'
print(f'✅ JAX detected {gpu_count} GPU(s).')
if gpu_count < 2:
    print('⚠️ Expected 2x T4 but JAX only sees', gpu_count, '— check accelerator setting.')

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
echo "🎉 Kaggle GPU T4 x2 Setup Completed!"
echo "NOTE: set this in every notebook cell that runs experiments:"
echo "  %env MUJOCO_GL=egl"
echo ""
echo "2 GPUs are visible as separate CUDA devices (0 and 1) — see"
echo "scripts/kaggle/README.md for how to run E1/E2 seeds in parallel,"
echo "one process per GPU via CUDA_VISIBLE_DEVICES."
echo "========================================================="
