# Experiment: exp_20260105_090955

## Goal
Train RL traffic signal control agent on 6x6 realistic grid with heavy traffic (22k vehicles).

## Key Parameters
- Grid: 6x6, realistic lanes, 2 junctions removed
- Vehicles: 22,000 (100% passenger, inner routes only)
- Reward: empirical with throughput_bonus=0.2
- Learning rate: 1e-4

## Training History
- **Started**: 2026-01-05
- **Checkpoint 491520** (2026-01-31): reward=110.3 (good baseline)
- **Checkpoint 495616** (2026-02-06): reward=111.2 (current best)

## Critical Findings
- `throughput_bonus` must be 0.2 (not default 0.1) - training collapses otherwise
- `learning_rate` must be 1e-4 (not 3e-4) - must match original training

## Results
- Total steps: 495,616
- Best mean reward: 111.2
- Status: Ready to continue on TAU
