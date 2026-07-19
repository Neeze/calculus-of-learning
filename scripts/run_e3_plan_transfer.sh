#!/bin/bash
set -e

echo "=== Running E3: Plan Transfer Across Frames ==="
export PYTHONPATH=$PYTHONPATH:$(pwd)

ENV="cartpole-swingup"
MAPPING="orthogonal" # identity, orthogonal, affine, mlp

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --mapping) MAPPING="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Environment: $ENV"
echo "Mapping type: $MAPPING"

    uv run python experiments/e3_plan_transfer/transfer_plan.py --env $ENV
