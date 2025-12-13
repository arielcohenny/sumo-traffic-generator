#!/bin/bash

#
# Batch data collection script for empirical reward function design
#
# This script runs Tree Method vs Fixed timing comparisons to collect comprehensive
# traffic metrics every 90 seconds. The data will be analyzed to identify which
# metrics discriminate good from bad performance.
#
# Usage: ./collect_comparative_data.sh
#
# Expected runtime: ~100 minutes (10 runs × ~10 min each)
#

set -e  # Exit on error

# Configuration
NUM_EPISODES=5
OUTPUT_DIR="evaluation/comparative_analysis"
BASE_CMD="env PYTHONUNBUFFERED=1 python -m src.cli"

# Fixed parameters (from user's specification)
NETWORK_SEED=24208
GRID_DIM=6
JUNCTIONS_REMOVE=2
BLOCK_SIZE=280
LANE_COUNT="realistic"
STEP_LENGTH=1.0
LAND_USE_BLOCK_SIZE=25.0
ATTRACTIVENESS="land_use"
TRAFFIC_LIGHT_STRATEGY="partial_opposites"
ROUTING_STRATEGY="realtime 100"
VEHICLE_TYPES="passenger 100"
PASSENGER_ROUTES="in 0 out 0 inner 100 pass 0"
DEPARTURE_PATTERN="uniform"
START_TIME_HOUR=8.0
NUM_VEHICLES=22000
END_TIME=7300

# Seed pairs for 5 episodes (different private and public traffic patterns)
SEED_PAIRS=(
    "72632 27031"
    "18475 93026"
    "54231 41829"
    "69873 12457"
    "31546 78902"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Comparative Data Collection"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Episodes: $NUM_EPISODES"
echo "  Methods: tree_method, fixed"
echo "  Duration: 2 hours simulation time (7300s)"
echo "  Metrics: Collected every 90 seconds"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Expected runtime: ~100 minutes (10 runs × ~10 min each)"
echo ""
echo "========================================="
echo ""

# Function to run a single simulation
run_simulation() {
    local episode=$1
    local method=$2
    local private_seed=$3
    local public_seed=$4
    local metric_log="${OUTPUT_DIR}/episode_${episode}_${method}_metrics.csv"
    local run_log="${OUTPUT_DIR}/episode_${episode}_${method}_run.log"

    echo "Episode ${episode}: Running ${method} (seeds: private=${private_seed}, public=${public_seed})..."
    echo "  Metric log: $metric_log"
    echo "  Run log: $run_log"

    # Build and run command
    $BASE_CMD \
        --network-seed $NETWORK_SEED \
        --grid_dimension $GRID_DIM \
        --junctions_to_remove $JUNCTIONS_REMOVE \
        --block_size_m $BLOCK_SIZE \
        --lane_count "$LANE_COUNT" \
        --step-length $STEP_LENGTH \
        --land_use_block_size_m $LAND_USE_BLOCK_SIZE \
        --attractiveness "$ATTRACTIVENESS" \
        --traffic_light_strategy "$TRAFFIC_LIGHT_STRATEGY" \
        --routing_strategy "$ROUTING_STRATEGY" \
        --vehicle_types "$VEHICLE_TYPES" \
        --passenger-routes "$PASSENGER_ROUTES" \
        --departure_pattern "$DEPARTURE_PATTERN" \
        --private-traffic-seed $private_seed \
        --public-traffic-seed $public_seed \
        --start_time_hour $START_TIME_HOUR \
        --num_vehicles $NUM_VEHICLES \
        --end-time $END_TIME \
        --traffic_control "$method" \
        --metric-log-path "$metric_log" \
        2>&1 | tee "$run_log"

    # Extract final performance metrics
    local throughput=$(grep "Vehicles arrived:" "$run_log" | tail -1 | awk '{print $3}' | tr -d ',')
    local avg_duration=$(grep "Average duration:" "$run_log" | tail -1 | awk '{print $3}' | tr -d 's')

    echo "  Throughput: $throughput vehicles"
    echo "  Avg Duration: ${avg_duration}s"
    echo "  Status: Complete"
    echo ""

    # Append summary to results file
    echo "${episode},${method},${private_seed},${public_seed},${throughput},${avg_duration}" >> "${OUTPUT_DIR}/summary.csv"
}

# Initialize summary file
echo "episode,method,private_seed,public_seed,throughput,avg_duration" > "${OUTPUT_DIR}/summary.csv"

# Main execution loop
start_time=$(date +%s)

for episode in $(seq 1 $NUM_EPISODES); do
    # Parse seed pair
    seed_pair=(${SEED_PAIRS[$((episode-1))]})
    private_seed=${seed_pair[0]}
    public_seed=${seed_pair[1]}

    echo "----------------------------------------"
    echo "Episode $episode / $NUM_EPISODES"
    echo "Private seed: $private_seed"
    echo "Public seed: $public_seed"
    echo "----------------------------------------"
    echo ""

    # Run tree_method
    run_simulation $episode "tree_method" $private_seed $public_seed

    # Run fixed
    run_simulation $episode "fixed" $private_seed $public_seed
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))

echo "========================================="
echo "Data Collection Complete!"
echo "========================================="
echo ""
echo "Total runs: $((NUM_EPISODES * 2))"
echo "Elapsed time: ${elapsed}s (~${elapsed_min} minutes)"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - summary.csv: Performance summary for all runs"
echo "  - episode_*_metrics.csv: Detailed metrics every 90s"
echo "  - episode_*_run.log: Full simulation logs"
echo ""
echo "Next steps:"
echo "  1. Review summary.csv to verify good vs bad runs"
echo "  2. Run analysis script: python scripts/analyze_comparative_metrics.py"
echo "  3. Design reward function based on discriminative metrics"
echo ""
