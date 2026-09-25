#!/usr/bin/env bash
set -euo pipefail
ALLEN_PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALLEN_ROOT="${ALLEN_ROOT:-$HOME/Allen1_CellTransformer2}"
mkdir -p "$ALLEN_ROOT/logs"
ALLEN_LOG="$ALLEN_ROOT/logs/full_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash "$ALLEN_PACKAGE/run_all.sh" --root "$ALLEN_ROOT" --epochs "${ALLEN_EPOCHS:-10}" "$@" \
  > "$ALLEN_LOG" 2>&1 < /dev/null &
ALLEN_PID=$!
printf 'Storage: %s\nPID: %s\nLog: %s\n' "$ALLEN_ROOT" "$ALLEN_PID" "$ALLEN_LOG"
printf "Follow: tail -f '%s'\n" "$ALLEN_LOG"
echo 'The process runs on the remote workstation after you disconnect. A launch PID alone does not mean setup passed.'
