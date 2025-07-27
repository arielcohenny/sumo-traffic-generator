#!/bin/bash

echo "🧪 TESTING ARGUMENT PARSING"
echo "=========================="

cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator
source .venv/bin/activate

echo "✅ Testing with properly quoted arguments..."

# Test the exact command with proper argument structure
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 400 \
  --vehicle_types "passenger 60 commercial 30 public 10" \
  --routing_strategy "shortest 80 realtime 20" \
  --departure_pattern six_periods \
  --end-time 7300 \
  --junctions_to_remove 0 \
  --lane_count realistic \
  --attractiveness poisson \
  --step-length 1.0 \
  --time_dependent \
  --seed 42 \
  --traffic_control tree_method

echo "🏁 Test completed!"