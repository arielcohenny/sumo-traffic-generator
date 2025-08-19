# SUMO Traffic Generator Examples

## Tree Method Dataset Examples

### High Traffic Load (Realistic)

Runs pre-built research network with high traffic density using Tree Method algorithm.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/decentralized_traffic_bottleneck/Experiment1-realistic-high-load/1 --traffic_control tree_method --end-time 7300 --gui
```

### High Traffic Load (Random)

Tests Tree Method performance on randomized high-density traffic patterns.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/decentralized_traffic_bottleneck/Experiment2-rand-high-load/1 --traffic_control tree_method --end-time 7300 --gui
```

### Moderate Traffic Load (Realistic)

Validates Tree Method on moderate traffic with realistic flow patterns.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/decentralized_traffic_bottleneck/Experiment3-realistic-moderate-load/1 --traffic_control tree_method --end-time 7300 --gui
```

## Synthetic Grid Examples

### Basic Grid

Simple 5x5 grid simulation for quick testing.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --end-time 1800 --gui
```

### Basic Grid with custom lanes

Simple 5x5 grid with custom lanes simulation for quick testing.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --end-time 1800 --gui --custom_lanes A1B1=head:B1C1:3,B1B2:1,B1B0:1;
```

### High Density Test

Stress test with 1200 vehicles and mixed routing strategies.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 6 --num_vehicles 1200 --end-time 3600 --routing_strategy 'shortest 60 realtime 40' --traffic_control actuated --gui
```

### Infrastructure Test

Network resilience test with removed junctions and realistic lanes.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 6 --junctions_to_remove 2 --num_vehicles 1000 --lane_count realistic --end-time 5400 --gui
```

### Rush Hour Simulation

Morning rush hour pattern with Tree Method optimization.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 200 --num_vehicles 2000 --departure_pattern six_periods --start_time_hour 7.0 --traffic_control tree_method --end-time 3600 --gui
```

### Multi-Modal Traffic

Mixed vehicle types with land use attractiveness model.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --block_size_m 150 --num_vehicles 850 --vehicle_types 'passenger 50 commercial 40 public 10' --attractiveness land_use --gui
```

### Advanced Routing

Three-strategy routing mix with time-dependent traffic patterns.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 4 --num_vehicles 600 --routing_strategy 'shortest 40 realtime 30 fastest 30' --time_dependent --end-time 3600 --gui
```

### Extended Simulation

Full 24-hour city simulation with temporal traffic patterns.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 7500 --end-time 86400 --departure_pattern six_periods --time_dependent --gui
```

### Weekend Pattern

Low-density weekend traffic with uniform departure timing.

```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 400 --departure_pattern uniform --start_time_hour 10.0 --attractiveness hybrid --end-time 7200 --gui
```

