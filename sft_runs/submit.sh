#!/bin/bash
# Detach launcher — fire and forget.
# Usage:
#   bash /mnt/bn/ecom-govern-models/qijiahe/sft_runs/submit.sh router_arch_a_3cls
#
# Spawns the named run (must match an existing router_*.sh) under nohup,
# redirects all output to outputs/<run>/train.log, writes pid file, returns.
# Tail with: tail -f /mnt/bn/ecom-govern-models/qijiahe/sft_runs/outputs/<run>/train.log

set -euo pipefail

RUN="${1:?usage: submit.sh <run_name>  e.g. router_arch_a_3cls}"
RUNS_DIR=/mnt/bn/ecom-govern-models/qijiahe/sft_runs
SCRIPT="${RUNS_DIR}/${RUN}.sh"
OUT_DIR="${RUNS_DIR}/outputs/${RUN}"
LOG="${OUT_DIR}/train.log"
PID_FILE="${OUT_DIR}/train.pid"

[ -f "${SCRIPT}" ] || { echo "[submit] no such run script: ${SCRIPT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

# refuse to launch if a previous pid is still alive
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "[submit] ${RUN} already running (pid $(cat "${PID_FILE}"))" >&2
  exit 1
fi

nohup bash "${SCRIPT}" >"${LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"
disown "${PID}" 2>/dev/null || true

echo "[submit] launched ${RUN} pid=${PID}"
echo "[submit] log  : ${LOG}"
echo "[submit] pid  : ${PID_FILE}"
echo "[submit] tail : tail -f ${LOG}"
echo "[submit] stop : kill \$(cat ${PID_FILE})"
