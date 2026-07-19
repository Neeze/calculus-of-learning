#!/bin/bash
set -e

echo "=== Running E2: Frame Freedom vs OOD ==="
export PYTHONPATH=$PYTHONPATH:$(pwd)

ENV="walker-walk"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Environment: $ENV"

# E2 requires training models with at least 2 seeds for each regularizer
REGULARIZERS=("full_rec" "light_rec" "vicreg" "light_vicreg" "sigreg")
SEEDS=(42 43)

echo "Training E2 models..."
for reg in "${REGULARIZERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Training $reg with seed $seed..."
        uv run python experiments/e2_frame_freedom/train_e2.py --env "$ENV" --reg "$reg" --seed "$seed"
    done
done

echo "Evaluating E2..."
uv run python experiments/e2_frame_freedom/eval_e2.py --env "$ENV"

echo "E2 flow complete."
