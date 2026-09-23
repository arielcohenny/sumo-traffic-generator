#!/usr/bin/env bash
# Run one simulation of the paper scenario with the frozen app copy in ../app.
#
# Usage:  scripts/run.sh <tm|rl> <run_name>
# Output: results/runs/<run_name>/workspace/   (SUMO outputs)
#         results/runs/<run_name>/run.log      (full console log)
#
# Run from anywhere; paths are resolved relative to this script.

set -euo pipefail

CONTROLLER="${1:?usage: run.sh <tm|rl> <run_name>}"
RUN_NAME="${2:?usage: run.sh <tm|rl> <run_name>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/app"
RUN_DIR="$ROOT/results/runs/$RUN_NAME"
MODEL="$ROOT/model/rl_traffic_model_1945600_steps.zip"

# Scenario: exp_20260105_090955 (network 24208, private 72632, public 27031)
SCENARIO=(
  --network-seed 24208 --grid_dimension 6 --junctions_to_remove 2 --block_size_m 280
  --lane_count realistic --step-length 1.0 --land_use_block_size_m 25.0 --attractiveness land_use
  --traffic_light_strategy partial_opposites --end-time 7300 --num_vehicles 22000
  --private-traffic-seed 72632 --public-traffic-seed 27031
  --routing_strategy "realtime 100" --vehicle_types "passenger 100"
  --passenger-routes "in 0 out 0 inner 100 pass 0" --departure_pattern uniform --start_time_hour 8.0
)

case "$CONTROLLER" in
  tm) CONTROL=(--traffic_control tree_method) ;;
  rl) CONTROL=(--traffic_control rl --rl_model_path "$MODEL" --rl-cycle-lengths 90) ;;
  *) echo "controller must be 'tm' or 'rl'" >&2; exit 1 ;;
esac

mkdir -p "$RUN_DIR"
cd "$APP"

# Guard: the app copy must be the one imported (not the repo-root src/ via an editable install)
SRC_FILE="$(python -c 'import src, os; print(os.path.realpath(src.__file__))')"
case "$SRC_FILE" in
  "$(realpath "$APP")"/src/*) ;;
  *) echo "ERROR: python imports src from $SRC_FILE, expected $APP/src" >&2; exit 1 ;;
esac

{
  echo "run_name: $RUN_NAME   controller: $CONTROLLER   started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "src: $SRC_FILE"
  echo "python: $(python --version 2>&1)"
  echo "sumo: $(sumo --version 2>&1 | head -1)"
  python -c "import numpy, torch, stable_baselines3 as s; print('numpy', numpy.__version__, 'torch', torch.__version__, 'sb3', s.__version__)" 2>/dev/null
  if [ "$CONTROLLER" = rl ]; then echo "model md5: $(python -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$MODEL")"; fi
} > "$RUN_DIR/run.log"

env PYTHONUNBUFFERED=1 python -m src.cli "${SCENARIO[@]}" "${CONTROL[@]}" --workspace "$RUN_DIR" >> "$RUN_DIR/run.log" 2>&1

grep -E "Throughput:|Average duration:" "$RUN_DIR/run.log"
