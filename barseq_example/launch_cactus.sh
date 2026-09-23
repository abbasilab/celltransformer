#!/usr/bin/env bash
set -euo pipefail
BARSEQ_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BARSEQ_ROOT="${BARSEQ_ROOT:-$HOME/BARseq_CT2_example}"
mkdir -p "$BARSEQ_ROOT/logs"
BARSEQ_LOG="$BARSEQ_ROOT/logs/full_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash "$BARSEQ_PACKAGE/run_all.sh" --root "$BARSEQ_ROOT" --epochs "${BARSEQ_EPOCHS:-10}" \
  > "$BARSEQ_LOG" 2>&1 < /dev/null &
BARSEQ_PID=$!
printf 'Storage: %s\nPID: %s\nLog: %s\n' "$BARSEQ_ROOT" "$BARSEQ_PID" "$BARSEQ_LOG"
printf "Follow: tail -f '%s'\n" "$BARSEQ_LOG"
echo 'The process runs on cactus after you disconnect. A launch PID alone does not mean setup passed.'
