#!/usr/bin/env bash
set -euo pipefail
BARSEQ_ROOT="${BARSEQ_ROOT:-$HOME/BARseq_CT2_example}"
if [[ ! -f "$BARSEQ_ROOT/LATEST_BARSEQ_CT2" ]]; then
  echo 'No run pointer yet. Inspect the full_*.log file printed by the launcher.'
  exit 1
fi
BARSEQ_RUN="$(cat "$BARSEQ_ROOT/LATEST_BARSEQ_CT2")"
echo "Run: $BARSEQ_RUN"
cat "$BARSEQ_RUN/status.json"
if [[ -f "$BARSEQ_RUN/PID" ]]; then
  BARSEQ_PID="$(cat "$BARSEQ_RUN/PID")"
  ps -p "$BARSEQ_PID" -o pid,etime,pcpu,pmem,args || true
fi
if [[ -f "$BARSEQ_RUN/SUCCESS" ]]; then
  echo 'ALL STAGES COMPLETED:'
  cat "$BARSEQ_RUN/SUCCESS"
  ls -lh "$BARSEQ_RUN/report/review.html" "$BARSEQ_RUN/report/results.zip"
else
  echo 'The full pipeline has not completed. Passed training alone does not mean the report is ready.'
fi
