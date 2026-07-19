#!/bin/bash
set -e

echo "=== Running E2: Frame Freedom vs OOD ==="
export PYTHONPATH=$PYTHONPATH:$(pwd)

ENV="walker-walk"
REGULARIZER="jepa" # full_rec, light_rec, vicreg, light_vicreg, jepa

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --reg) REGULARIZER="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Environment: $ENV"
echo "Regularizer: $REGULARIZER"

# uv run python experiments/e2_frame_freedom/run.py --env $ENV --reg $REGULARIZER

echo "To be implemented..."
