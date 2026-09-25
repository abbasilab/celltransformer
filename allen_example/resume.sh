#!/usr/bin/env bash
set -euo pipefail
ALLEN_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALLEN_ROOT="${ALLEN_ROOT:-$HOME/Allen1_CellTransformer2}"
ALLEN_RUN="${1:-$(cat "$ALLEN_ROOT/LATEST_ALLEN_CT2")}"
mkdir -p "$ALLEN_ROOT/logs"
ALLEN_LOG="$ALLEN_ROOT/logs/resume_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash "$ALLEN_PACKAGE/run_all.sh" --resume "$ALLEN_RUN" > "$ALLEN_LOG" 2>&1 < /dev/null &
printf 'PID: %s\nLog: %s\n' "$!" "$ALLEN_LOG"
printf "Follow: tail -f '%s'\n" "$ALLEN_LOG"
