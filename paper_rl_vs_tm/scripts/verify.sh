#!/usr/bin/env bash
# Check that a run reproduces the paper's reference numbers exactly (SUMO 1.25.0).
#
# Usage:  scripts/verify.sh <tm|rl> <run_name>

set -euo pipefail

CONTROLLER="${1:?usage: verify.sh <tm|rl> <run_name>}"
RUN_NAME="${2:?usage: verify.sh <tm|rl> <run_name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/results/runs/$RUN_NAME/run.log"

case "$CONTROLLER" in
  tm) EXPECTED_THROUGHPUT=8194; EXPECTED_DURATION=727.0 ;;
  rl) EXPECTED_THROUGHPUT=8660; EXPECTED_DURATION=577.5 ;;
  *) echo "controller must be 'tm' or 'rl'" >&2; exit 1 ;;
esac

THROUGHPUT="$(grep -oE 'Throughput: [0-9]+' "$LOG" | awk '{print $2}')"
DURATION="$(grep -oE 'Average duration: [0-9.]+' "$LOG" | awk '{print $3}')"

echo "$RUN_NAME ($CONTROLLER): throughput $THROUGHPUT (expected $EXPECTED_THROUGHPUT), duration $DURATION (expected $EXPECTED_DURATION)"
if [ "$THROUGHPUT" = "$EXPECTED_THROUGHPUT" ] && [ "$DURATION" = "$EXPECTED_DURATION" ]; then
  echo "REPRODUCED"
else
  echo "NOT REPRODUCED"
  exit 1
fi
