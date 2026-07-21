#!/usr/bin/env bash
# APPS functional-style rollout — separate from apps.jsonl (stdin-style).
# Output: outputs/rollouts/v55/apps_functional.jsonl  (+ .pretty.txt)
#
# These are the ~2169 APPS train problems with fn_name set (LeetCode-ish but
# without class Solution wrappers). Tests run via apps_functional test_format,
# which synthesizes a call-compat harness around the model's function.

set -uo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
export PYTHONPATH=.

OUTDIR=outputs/rollouts/v55
LOGDIR=$OUTDIR/logs
mkdir -p "$OUTDIR" "$LOGDIR"

LOG=$LOGDIR/apps_functional.log
DONE=$LOGDIR/APPS_FUNCTIONAL.DONE

CONCURRENCY=${CONCURRENCY:-4}
LIMIT=${LIMIT:-3000}            # ~2169 functional; +buffer

ts() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo
  echo "[$(ts)] === START APPS-functional rollout → $OUTDIR/apps_functional.jsonl ==="
  echo "  concurrency=$CONCURRENCY  limit=$LIMIT"
} >> "$LOG"

set +e
python -u scripts/collect_rollouts_opt.py \
    --dataset apps_functional \
    --limit "$LIMIT" \
    --concurrency "$CONCURRENCY" \
    --base_model gpt-5.4-nano --escalate_to gpt-5.4 \
    --dataset_tag apps_functional \
    --out "$OUTDIR/apps_functional.jsonl" \
    >> "$LOG" 2>&1
rc=$?
set -e
echo "[$(ts)] === END APPS-functional (exit $rc, $(wc -l < $OUTDIR/apps_functional.jsonl 2>/dev/null || echo 0) records) ===" >> "$LOG"
touch "$DONE"
