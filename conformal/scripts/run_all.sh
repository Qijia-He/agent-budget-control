#!/bin/bash
# End-to-end CRC pipeline: data split -> predict -> calibrate -> evaluate.
# Run from anywhere; uses absolute paths.

set -euo pipefail

CONF=/mnt/bn/ecom-govern-models/qijiahe/conformal

# activate llamafactory env (has torch+transformers+peft)
source /mnt/bn/ecom-govern-models/qijiahe/setup_env.sh

cd "${CONF}"

echo "===== 1. split data 80/10/10 ====="
python scripts/data_split.py

echo ""
echo "===== 2. predict on calibration split ====="
python scripts/router_predict.py \
  --input  "${CONF}/data/calib.jsonl" \
  --output "${CONF}/data/calib_preds.jsonl"

echo ""
echo "===== 3. predict on test split ====="
python scripts/router_predict.py \
  --input  "${CONF}/data/test.jsonl" \
  --output "${CONF}/data/test_preds.jsonl"

echo ""
echo "===== 4. calibrate tau for alpha sweep ====="
python scripts/calibrate_crc.py

echo ""
echo "===== 5. evaluate CRC on test split ====="
python scripts/eval_crc.py

echo ""
echo "===== DONE — see ${CONF}/results/eval.json and FINDINGS.md ====="
