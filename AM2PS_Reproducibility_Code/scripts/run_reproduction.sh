#!/usr/bin/env bash
set -euo pipefail

# Full manuscript-aligned reproduction sequence.
# Dataset paths must match the YAML files before this script is run.

python scripts/validate_dataset.py data/ChestXRay2017/chest_xray
python scripts/validate_dataset.py data/ChestXRay-Covid19

python tests/smoke_test.py

python train.py --config configs/chestxray2017.yaml --device cuda
python train.py --config configs/chestxray_covid19.yaml --device cuda

python experiments/run_ablation.py \
  --config configs/chestxray2017.yaml \
  --output-root outputs/ablation_chestxray2017 \
  --device cuda
python experiments/collect_results.py outputs/ablation_chestxray2017 \
  --output outputs/ablation_chestxray2017.csv

python experiments/run_prompt_sweep.py \
  --config configs/chestxray2017.yaml \
  --output-root outputs/prompt_sweep \
  --device cuda
python experiments/run_scale_layer_sweep.py \
  --config configs/chestxray2017.yaml \
  --output-root outputs/scale_layer_sweep \
  --device cuda
python experiments/run_lambda_sweep.py \
  --config configs/chestxray2017.yaml \
  --output-root outputs/lambda_sweep \
  --device cuda
