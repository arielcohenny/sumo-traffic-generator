#!/bin/bash

# Batch download script for original research results
# Downloads all algorithm results from the 80 test cases

BASE_URL="https://raw.githubusercontent.com/nimrodSerokTAU/decentralized-traffic-bottlenecks/main/data"
BASE_DIR="evaluation/datasets/decentralized_traffic_bottleneck"

# Create validation baselines directory
mkdir -p evaluation/validation/baselines

echo "Starting download of all original research results..."

# Download main results file
echo "Downloading main results.xlsx..."
curl -s "$BASE_URL/results.xlsx" > evaluation/validation/baselines/original_results.xlsx

# List of experiments
experiments=(
    "Experiment1-realistic-high-load"
    "Experiment2-rand-high-load" 
    "Experiment3-realistic-moderate-load"
    "Experiment4-and-moderate-load"
)

# Algorithm files to download
declare -A algorithm_files=(
    ["tree_method_driving_times.txt"]="CurrentTreeDvd/driving_time_distribution.txt"
    ["tree_method_vehicle_stats.txt"]="CurrentTreeDvd/vehicles_stats.txt"
    ["tree_method_costs.txt"]="CurrentTreeDvd/tree_cost_distribution.txt"
    ["actuated_driving_times.txt"]="SUMOActuated/driving_time_distribution.txt"
    ["actuated_vehicle_stats.txt"]="SUMOActuated/vehicles_stats.txt"
    ["actuated_costs.txt"]="SUMOActuated/tree_cost_distribution.txt"
    ["random_driving_times.txt"]="Random/driving_time_distribution.txt"
    ["random_vehicle_stats.txt"]="Random/vehicles_stats.txt"
    ["uniform_driving_times.txt"]="Uniform/driving_time_distribution.txt"
    ["uniform_vehicle_stats.txt"]="Uniform/vehicles_stats.txt"
)

successful_downloads=0
total_files=0

# Download results for each experiment
for experiment in "${experiments[@]}"; do
    echo "Processing $experiment..."
    
    # Download experiment summary files
    for summary_file in "res_ended_vehicles_count.txt" "res_time per v.txt"; do
        echo "  Downloading experiment summary: $summary_file"
        curl -s "$BASE_URL/$experiment/$summary_file" > "evaluation/validation/baselines/${experiment}_${summary_file}"
        ((total_files++))
        if [ -s "evaluation/validation/baselines/${experiment}_${summary_file}" ]; then
            ((successful_downloads++))
        fi
        sleep 0.1
    done
    
    # Download test case results (cases 1-20)
    for case_num in {1..20}; do
        case_dir="$BASE_DIR/$experiment/$case_num/original_results"
        mkdir -p "$case_dir"
        
        echo "  Downloading test case $case_num..."
        
        # Download each algorithm result file
        for target_file in "${!algorithm_files[@]}"; do
            source_file="${algorithm_files[$target_file]}"
            url="$BASE_URL/$experiment/$case_num/$source_file"
            output_path="$case_dir/$target_file"
            
            curl -s "$url" > "$output_path"
            ((total_files++))
            
            # Check if download was successful (file not empty and doesn't contain "404")
            if [ -s "$output_path" ] && ! grep -q "404" "$output_path"; then
                ((successful_downloads++))
            else
                # Remove empty or 404 files
                rm -f "$output_path"
            fi
            
            # Small delay to be respectful
            sleep 0.05
        done
    done
    
    success_rate=$(( (successful_downloads * 100) / total_files ))
    echo "  Completed $experiment: $successful_downloads/$total_files files ($success_rate%)"
done

echo ""
echo "Download Summary:"
echo "- Total files attempted: $total_files"
echo "- Successful downloads: $successful_downloads" 
echo "- Success rate: $(( (successful_downloads * 100) / total_files ))%"
echo ""
echo "Files saved to:"
echo "- Main results: evaluation/validation/baselines/original_results.xlsx"
echo "- Test case results: $BASE_DIR/*/original_results/"
echo "- Experiment summaries: evaluation/validation/baselines/"