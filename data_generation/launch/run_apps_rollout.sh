#!/usr/bin/env bash
# APPS large-scale rollout — same format as Phase A bcb/lcb/taco_*.
# Output: outputs/rollouts/v55/apps.jsonl  (+ .pretty.txt)
#
# Probe showed APPS has the best cascade profile we've found:
#   A=50%, B3=83% (+33pp), cheap-recovery niche 23%, escalate niche 17%,
#   unsolvable only 10%. Cheapest-label distribution 50/20/3/17/10.
#
# Launch:
#   tmux new-session -d -s confo-apps "bash /cloudide/workspace/conformal_react/code/run_apps_rollout.sh"
# Progress:
#   tail -f outputs/rollouts/v55/logs/apps.log
# Resume: re-running skips already-done problem_ids.

set -uo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
export PYTHONPATH=.
export LCB_MAX_TEST_CASES=20

OUTDIR=outputs/rollouts/v55
LOGDIR=$OUTDIR/logs
mkdir -p "$OUTDIR" "$LOGDIR"

LOG=$LOGDIR/apps.log
DONE=$LOGDIR/APPS.DONE

CONCURRENCY=${CONCURRENCY:-4}   # Phase A's safe value; conc=8 saturates QPM ceiling
LIMIT=${LIMIT:-8000}            # APPS test+train (stdin-only) = 6698 problems; +buffer

ts() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo
  echo "[$(ts)] === START APPS rollout → $OUTDIR/apps.jsonl ==="
  echo "  concurrency=$CONCURRENCY  limit=$LIMIT"
  echo "  base=gpt-5.4-nano  escalate=gpt-5.4"
} >> "$LOG"

set +e
python -u scripts/collect_rollouts_opt.py \
    --dataset apps \
    --limit "$LIMIT" \
    --concurrency "$CONCURRENCY" \
    --base_model gpt-5.4-nano --escalate_to gpt-5.4 \
    --dataset_tag apps \
    --out "$OUTDIR/apps.jsonl" \
    >> "$LOG" 2>&1
rc=$?
set -e
echo "[$(ts)] === END APPS rollout (exit $rc, $(wc -l < $OUTDIR/apps.jsonl 2>/dev/null || echo 0) records) ===" >> "$LOG"
touch "$DONE"
