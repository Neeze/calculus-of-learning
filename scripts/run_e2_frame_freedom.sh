#!/bin/bash
set -e

echo "=== Running E2: Frame Freedom vs OOD ==="
export PYTHONPATH=$PYTHONPATH:$(pwd)

ENV="walker-walk"
SKIP_SANITY=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --skip-sanity) SKIP_SANITY=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Environment: $ENV"

# Spec §3.2-3.3: 5 configs x 3 seeds, shared cached datasets per seed.
REGULARIZERS=("full_rec" "light_rec" "vicreg" "light_vicreg" "sigreg")
SEEDS=(0 1 2)

echo "[1/5] Collecting cached datasets (train per seed + eval ID/OOD grid)..."
uv run python experiments/e2_frame_freedom/collect_e2.py --env "$ENV" --train_seeds "${SEEDS[@]}"

if [[ "$SKIP_SANITY" -eq 0 ]]; then
    echo "[2/5] Shift-strength oracle (spec §6.2) — check output before trusting the sweep..."
    uv run python experiments/e2_frame_freedom/sanity_shift_e2.py --env "$ENV"
else
    echo "[2/5] Skipping sanity check (--skip-sanity)"
fi

echo "[3/5] Training 5 configs x 3 seeds..."
for reg in "${REGULARIZERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Training $reg with seed $seed..."
        uv run python experiments/e2_frame_freedom/train_e2.py --env "$ENV" --reg "$reg" --seed "$seed"
    done
done

echo "[4/5] Evaluating (raw metrics -> results.json)..."
uv run python experiments/e2_frame_freedom/eval_e2.py --env "$ENV" --seeds "${SEEDS[@]}"

echo "[5/5] Analysis (statistics + figures)..."
uv run python experiments/e2_frame_freedom/analyze_e2.py --env "$ENV"

echo "E2 flow complete. Results in outputs/results/e2/, plots in outputs/plots/."
