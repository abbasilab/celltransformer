#!/usr/bin/env bash
set -euo pipefail
ALLEN_ROOT="${ALLEN_ROOT:-$HOME/Allen1_CellTransformer2}"
if [[ ! -f "$ALLEN_ROOT/LATEST_ALLEN_CT2" ]]; then
  echo 'No run pointer yet. Inspect the full_*.log file printed by the launcher.'
  exit 1
fi
ALLEN_RUN="$(cat "$ALLEN_ROOT/LATEST_ALLEN_CT2")"
echo "Run: $ALLEN_RUN"
cat "$ALLEN_RUN/status.json"
if [[ -f "$ALLEN_RUN/PID" ]]; then
  ALLEN_PID="$(cat "$ALLEN_RUN/PID")"
  ps -p "$ALLEN_PID" -o pid,etime,pcpu,pmem,args || true
fi
if [[ -f "$ALLEN_RUN/SUCCESS" ]]; then
  echo 'ALL STAGES COMPLETED:'
  cat "$ALLEN_RUN/SUCCESS"
  ls -lh "$ALLEN_RUN/report/review.html" "$ALLEN_RUN/report/results.zip"
else
  echo 'The full pipeline has not completed. Passed training alone does not mean the report is ready.'
fi
