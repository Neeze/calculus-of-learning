#!/bin/bash
# scripts/collab/setup_colab.sh
# Setup environment for Calculus of Learning on Google Colab GPU (T4/L4/A100)

set -e

echo "========================================================="
echo "🚀 Colab GPU Environment Setup for Calculus of Learning"
echo "========================================================="

# 1. Check GPU status
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️ WARNING: No GPU detected. Change Colab runtime type to T4/L4/A100,"
    echo "   or use scripts/collab/setup_colab_tpu.sh on a TPU runtime."
fi

# 2. System dependencies for headless rendering (package names vary across
#    Ubuntu versions — try new names first, fall back to old ones)
echo "📦 Installing system dependencies (EGL/OSMesa, FFmpeg)..."
sudo apt-get update -y
sudo apt-get install -y ffmpeg libosmesa6-dev patchelf libegl1 libgl1 > /dev/null \
    || sudo apt-get install -y ffmpeg libosmesa6-dev patchelf libgl1-mesa-glx > /dev/null

# 3. uv for fast dependency management
echo "⚡ Installing uv..."
pip install --upgrade -q uv
export UV_SYSTEM_PYTHON=1

# 4. Project dependencies
echo "🐍 Installing project dependencies..."
uv pip install --system -e .

# 5. DreamerV3 submodule (E1 Tier 1) — init it here so the script is
# self-sufficient regardless of what the notebook cells did.
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

# 6. Ensure CUDA jax wins — MUST be last install step (dreamerv3 requirements
#    may have downgraded or replaced it)
echo "🔥 Installing JAX with CUDA 12 support..."
uv pip install --system -U "jax[cuda12]>=0.6.2,<0.7" flax optax

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
assert any(d.platform == 'gpu' for d in devices), 'GPU not detected by JAX!'
print('✅ JAX successfully detected GPU.')

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
echo "🎉 GPU Setup Completed!"
echo "NOTE: set this in every notebook cell that runs experiments:"
echo "  %env MUJOCO_GL=egl"
echo "========================================================="
