#!/usr/bin/env bash
# Compress the SUMO outputs of one run into results/runs/<run_name>/outputs/*.xml.gz
# (the files kept in git; the rest of workspace/ is regenerated deterministically and not kept).
#
# Usage:  scripts/pack_outputs.sh <run_name>

set -euo pipefail

RUN_NAME="${1:?usage: pack_outputs.sh <run_name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/results/runs/$RUN_NAME"
WS="$RUN_DIR/workspace"

OUTPUTS=(tripinfo summary sumo_statistics tls_switches lanedata vehroutes grid.net)

mkdir -p "$RUN_DIR/outputs"
for name in "${OUTPUTS[@]}"; do
  gzip -9 -c "$WS/$name.xml" > "$RUN_DIR/outputs/$name.xml.gz"
done
ls -l "$RUN_DIR/outputs"
