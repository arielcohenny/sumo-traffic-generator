#!/bin/bash

# Master Script: Run All Decentralized Traffic Bottleneck Experiments
# Executes all 80 individual runs across 4 experiments
# WARNING: This will take many hours to complete!

set -e

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS=("Experiment1-realistic-high-load" "Experiment2-rand-high-load" "Experiment3-realistic-moderate-load" "Experiment4-and-moderate-load")

echo "DECENTRALIZED TRAFFIC BOTTLENECK MASTER EXECUTION"
echo "=================================================="
echo "This script will execute ALL 80 runs across 4 experiments:"
echo ""
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - $exp: 20 runs (Tree Method + Actuated + Fixed each)"
done
echo ""
echo "Total: 80 runs × 3 methods = 240 simulations"
echo "Expected time: 8-12 hours (depending on system performance)"
echo ""
echo "⚠️  WARNING: This is a very long-running process!"
echo "⚠️  Make sure you have sufficient disk space and time"
echo "⚠️  Consider running in a screen/tmux session"
echo ""

# Safety confirmation
read -p "Are you absolutely sure you want to run all 240 simulations? (yes/NO) " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled. Good choice! Consider running individual experiments first."
    echo ""
    echo "To run individual experiments:"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "  cd $exp && ./run_all_runs.sh"
    done
    exit 0
fi

# Ask about selective execution
echo "Execution options:"
echo "1) Run all 4 experiments sequentially (full benchmark)"
echo "2) Select specific experiments to run"
echo ""
read -p "Choose option (1 or 2): " -n 1 -r
echo

if [[ $REPLY =~ ^2$ ]]; then
    echo ""
    echo "Select experiments to run:"
    selected_experiments=()
    for i in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$i]}"
        read -p "Run $exp? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            selected_experiments+=("$exp")
        fi
    done
    
    if [ ${#selected_experiments[@]} -eq 0 ]; then
        echo "No experiments selected. Exiting."
        exit 0
    fi
    
    EXPERIMENTS=("${selected_experiments[@]}")
    echo "Selected experiments: ${EXPERIMENTS[*]}"
fi

echo ""
echo "Final confirmation: Running ${#EXPERIMENTS[@]} experiment(s)"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create master log
MASTER_LOG="$BENCHMARK_DIR/master_execution.log"
echo "Master execution started at $(date)" > "$MASTER_LOG"
echo "Selected experiments: ${EXPERIMENTS[*]}" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

start_time=$(date +%s)

echo ""
echo "🚀 Starting master execution at $(date)"
echo "📋 Master log: $MASTER_LOG"
echo ""

# Execute each selected experiment
for i in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))
    total_exp=${#EXPERIMENTS[@]}
    
    echo "==============================================="
    echo "EXPERIMENT $exp_num/$total_exp: $experiment"
    echo "==============================================="
    echo "Started at: $(date)"
    echo ""
    
    exp_start_time=$(date +%s)
    
    # Log experiment start
    {
        echo "========================================"
        echo "EXPERIMENT $exp_num/$total_exp: $experiment"
        echo "Started: $(date)"
        echo "========================================"
    } >> "$MASTER_LOG"
    
    # Change to experiment directory and run
    cd "$BENCHMARK_DIR/$experiment"
    
    if [ -f "run_all_runs.sh" ]; then
        # Run the experiment (automatically answer 'y' to confirmation)
        echo "y" | ./run_all_runs.sh 2>&1 | tee -a "$MASTER_LOG"
        
        exp_end_time=$(date +%s)
        exp_duration=$((exp_end_time - exp_start_time))
        exp_minutes=$((exp_duration / 60))
        exp_seconds=$((exp_duration % 60))
        
        echo ""
        echo "✅ Completed $experiment in ${exp_minutes}m ${exp_seconds}s"
        echo "Completed $experiment at $(date) (Duration: ${exp_minutes}m ${exp_seconds}s)" >> "$MASTER_LOG"
        echo "" >> "$MASTER_LOG"
        
        # Estimate remaining time
        if [ $exp_num -lt $total_exp ]; then
            avg_duration=$((($SECONDS) / exp_num))
            remaining_exp=$((total_exp - exp_num))
            est_remaining=$((avg_duration * remaining_exp))
            est_minutes=$((est_remaining / 60))
            est_hours=$((est_minutes / 60))
            est_minutes=$((est_minutes % 60))
            
            if [ $est_hours -gt 0 ]; then
                echo "⏱️  Estimated time remaining: ${est_hours}h ${est_minutes}m"
            else
                echo "⏱️  Estimated time remaining: ${est_minutes}m"
            fi
        fi
        
    else
        echo "❌ Error: run_all_runs.sh not found in $experiment"
        echo "ERROR: run_all_runs.sh not found in $experiment" >> "$MASTER_LOG"
    fi
    
    echo ""
    
    # Return to benchmark directory
    cd "$BENCHMARK_DIR"
done

end_time=$(date +%s)
total_duration=$((end_time - start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))

echo "================================================="
echo "🎉 MASTER EXECUTION COMPLETE!"
echo "================================================="
echo "Finished at: $(date)"
echo "Total duration: ${total_hours}h ${total_minutes}m ${total_seconds}s"
echo "Experiments completed: ${#EXPERIMENTS[@]}"
echo ""
echo "📊 Next steps:"
echo "  1. Run analysis: python3 analyze_all_experiments.py"
echo "  2. Check individual experiment analyses in each experiment folder"
echo "  3. Review master log: $MASTER_LOG"
echo ""

# Log completion
{
    echo "========================================"
    echo "MASTER EXECUTION COMPLETE"
    echo "Finished: $(date)"
    echo "Total duration: ${total_hours}h ${total_minutes}m ${total_seconds}s"
    echo "Experiments completed: ${#EXPERIMENTS[@]}"
    echo "========================================"
} >> "$MASTER_LOG"

echo "Master execution log saved to: $MASTER_LOG"
echo "🎯 All experiments complete! Ready for analysis."