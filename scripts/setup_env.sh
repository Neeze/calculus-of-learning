#!/bin/bash
set -e

echo "=== Setup Environment for Calculus of Learning ==="

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install uv first."
    exit 1
fi

echo "Installing base dependencies with uv..."
uv sync

echo "Installing DreamerV3 requirements..."
uv pip install -r third_party/dreamerv3/requirements.txt

echo "Installing specific ML packages (JAX, WandB, etc)..."
# Adding dependencies to pyproject.toml
uv add wandb flax optax tensorflow-cpu

# Note: jax[cuda] installation might depend on the specific CUDA version on the machine.
echo "Adding JAX with CUDA support..."
uv add "jax[cuda12]"

echo "Setup completed successfully."
