#!/bin/bash

BASE_URL="https://raw.githubusercontent.com/nimrodSerokTAU/decentralized-traffic-bottlenecks/main/data"
BASE_DIR="evaluation/datasets/decentralized_traffic_bottleneck"

echo "Starting download of all decentralized traffic bottleneck test cases..."

for experiment in "${experiments[@]}"; do
    echo "Processing $experiment..."
    
    # Create experiment directory
    mkdir -p "$BASE_DIR/$experiment"
    
    # Download test cases 1-20 for this experiment
    for case_num in {1..20}; do
        echo "  Downloading test case $case_num..."
        
        # Create case directory
        mkdir -p "$BASE_DIR/$experiment/$case_num"
        cd "$BASE_DIR/$experiment/$case_num"
        
        # Download each required file
        for file in "${files[@]}"; do
            echo "    Downloading $file..."
            curl -s "$BASE_URL/$experiment/$case_num/$file" > "$file"
            
            # Check if download was successful (file not empty)
            if [ ! -s "$file" ]; then
                echo "    WARNING: $file appears to be empty or failed to download"
            fi
        done
        
        # Return to project root
        cd - > /dev/null
    done
    
    echo "  Completed $experiment"
done

echo "Download complete! Summary:"
echo "- 4 experiments downloaded"
echo "- 20 test cases per experiment"  
echo "- 4 files per test case"
echo "- Total: 320 files downloaded"

# Verify download
echo ""
echo "Verification:"
for experiment in "${experiments[@]}"; do
    case_count=$(ls -1 "$BASE_DIR/$experiment" | wc -l)
    echo "- $experiment: $case_count test cases"
done