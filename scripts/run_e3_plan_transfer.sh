#!/bin/bash
set -e

echo "=== Running E3: Plan Transfer Across Frames ==="
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

# E3 relies on E2 checkpoints (specifically 'sigreg' with seeds 42 and 43)
# Make sure run_e2_frame_freedom.sh is executed first.

echo "Evaluating E3 plan transfer and MPC tracking..."
uv run python experiments/e3_plan_transfer/transfer_plan.py --env "$ENV"

echo "E3 flow complete."
