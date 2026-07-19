#!/bin/bash
set -e

echo "=== Running E1: Error Laws ==="
# Add python path so src can be imported
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default arguments
ENV="linear" # linear or pendulum
L_VALUE=1.05
SEEDS=5

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --l_value) L_VALUE="$2"; shift ;;
        --seeds) SEEDS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Environment: $ENV"
echo "L Value: $L_VALUE"
echo "Seeds: $SEEDS"

# Assuming the experiment python script will be created at experiments/e1_error_laws/run.py
if [ "$ENV" = "linear" ]; then
    echo "Running toy experiment with linear system..."
    uv run python experiments/e1_error_laws/train_toy.py --env $ENV --l_value $L_VALUE --seeds $SEEDS
    uv run python experiments/e1_error_laws/eval_toy.py --l_value $L_VALUE
else
    echo "DMC/Pendulum environment not yet implemented in run script."
fi
