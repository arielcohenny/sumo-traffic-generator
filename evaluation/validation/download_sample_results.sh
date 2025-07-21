#!/bin/bash

# Simple download script for a few original research results samples
BASE_URL="https://raw.githubusercontent.com/nimrodSerokTAU/decentralized-traffic-bottlenecks/main/data"

echo "Downloading sample original research results..."

# Create directories
mkdir -p evaluation/validation/baselines

# Download main results file
echo "Downloading main results.xlsx..."
curl -s "$BASE_URL/results.xlsx" > evaluation/validation/baselines/original_results.xlsx

# Download results for first few test cases of Experiment1
for case_num in 1 2 3; do
    echo "Downloading results for Experiment1 case $case_num..."
    
    case_dir="evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/$case_num/original_results"
    mkdir -p "$case_dir"
    
    # Tree Method results
    curl -s "$BASE_URL/Experiment1-realistic-high-load/$case_num/CurrentTreeDvd/driving_time_distribution.txt" > "$case_dir/tree_method_driving_times.txt"
    curl -s "$BASE_URL/Experiment1-realistic-high-load/$case_num/CurrentTreeDvd/vehicles_stats.txt" > "$case_dir/tree_method_vehicle_stats.txt"
    
    # SUMO Actuated results  
    curl -s "$BASE_URL/Experiment1-realistic-high-load/$case_num/SUMOActuated/driving_time_distribution.txt" > "$case_dir/actuated_driving_times.txt"
    curl -s "$BASE_URL/Experiment1-realistic-high-load/$case_num/SUMOActuated/vehicles_stats.txt" > "$case_dir/actuated_vehicle_stats.txt"
    
    sleep 0.5
done

echo "Sample download completed!"
echo "Downloaded results for Experiment1 test cases 1-3"