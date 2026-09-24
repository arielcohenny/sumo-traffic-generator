#!/usr/bin/env bash
# Run the SUMO-seed stability runs for the given seeds, one after another:
# for each seed, Tree Method (tm_sNN), RL (rl_sNN) and RL's fixed plan (fixed_sNN).
# A failed run is reported and the batch continues.
#
# Usage:  scripts/run_seed_batch.sh <seed> [<seed> ...]

set -uo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in "$@"; do
  nn="$(printf '%02d' "$seed")"
  SUMO_SEED="$seed" "$SCRIPTS/run.sh" tm "tm_s$nn" || echo "FAILED tm_s$nn"
  SUMO_SEED="$seed" "$SCRIPTS/run.sh" rl "rl_s$nn" || echo "FAILED rl_s$nn"
  SUMO_SEED="$seed" "$SCRIPTS/run.sh" rl "fixed_s$nn" results/plans/rl_modal.csv || echo "FAILED fixed_s$nn"
done
