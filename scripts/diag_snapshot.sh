#!/bin/bash
# Capture one-shot hang diagnostics for an XPU process.
#
#   scripts/diag_snapshot.sh <pid> [logfile] [outdir]
#
# Writes a timestamped snapshot: process state, xpu-smi, py-spy dump (if the
# GIL is released - attach to a native-spinning process fails, which is itself
# recorded), and the tail of the bench log. Safe to run while the device is
# hung; never kills anything.
set -u
PID=${1:?usage: diag_snapshot.sh <pid> [logfile] [outdir]}
LOG=${2:-}
OUT=${3:-/tmp/opencode/diagnostics}
mkdir -p "$OUT"
TS=$(date +%Y%m%d-%H%M%S)
F="$OUT/diag-$TS-pid$PID.txt"
{
  echo "=== diag snapshot $(date -Is) pid=$PID ==="
  echo "--- process ---"
  ps -o pid,ppid,stat,etime,pcpu,rss,cmd -p "$PID" 2>&1
  echo "--- xpu-smi ---"
  xpu-smi 2>&1 | sed -n '1,20p'
  echo "--- py-spy (may fail while native-spinning; GIL-released attach works) ---"
  py-spy dump --pid "$PID" 2>&1 | head -40
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    echo "--- log tail ($LOG) ---"
    tail -30 "$LOG"
    echo "--- log mtime age (s) ---"
    echo $(( $(date +%s) - $(stat -c %Y "$LOG") ))
  fi
  echo "--- env sanity ---"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
} > "$F" 2>&1
echo "wrote $F"
