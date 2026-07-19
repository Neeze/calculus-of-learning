#!/bin/bash
set -e

echo "=== Running E1: Error Laws ==="
# Add python path so src can be imported
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default arguments
ENV="linear" # linear or pendulum

echo "Environment: $ENV"

if [ "$ENV" = "linear" ]; then
    echo "Running toy experiment with linear system..."
    # The python scripts now handle sweeping over multiple L values and seeds internally.
    uv run python experiments/e1_error_laws/train_toy.py --l_values 0.8 0.95 1.05 1.2 --seeds 42 43 44
    uv run python experiments/e1_error_laws/eval_toy.py --l_values 0.8 0.95 1.05 1.2 --seeds 42 43 44
else
    echo "DMC/Pendulum environment not yet implemented in run script."
fi
