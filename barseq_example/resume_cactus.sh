#!/usr/bin/env bash
set -euo pipefail
BARSEQ_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BARSEQ_ROOT="${BARSEQ_ROOT:-$HOME/BARseq_CT2_example}"
BARSEQ_RUN="${1:-$(cat "$BARSEQ_ROOT/LATEST_BARSEQ_CT2")}"
mkdir -p "$BARSEQ_ROOT/logs"
BARSEQ_LOG="$BARSEQ_ROOT/logs/resume_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash "$BARSEQ_PACKAGE/run_all.sh" --resume "$BARSEQ_RUN" > "$BARSEQ_LOG" 2>&1 < /dev/null &
printf 'PID: %s\nLog: %s\n' "$!" "$BARSEQ_LOG"
printf "Follow: tail -f '%s'\n" "$BARSEQ_LOG"
