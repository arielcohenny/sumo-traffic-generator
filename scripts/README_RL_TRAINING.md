# RL Training Commands for Independent Execution

This document provides commands you can run independently to execute production-scale RL training without consuming Claude's tokens.

## Quick Start Commands

### Reliable Production Training (50k timesteps, single environment, frequent checkpoints)
```bash
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator
python scripts/train_rl_production.py --timesteps 50000 --single-env --checkpoint-freq 10000
```

### Chain Training for Higher Scale (multiple 50k runs)
```bash
# First run
python scripts/train_rl_production.py --timesteps 50000 --single-env --model-name chain1
# Continue training (when resume-from is implemented)
# python scripts/train_rl_production.py --timesteps 50000 --single-env --model-name chain2 --resume-from models/chain1_*.zip
```

### High-Scale Training (for systems with good SUMO stability)
```bash
python scripts/train_rl_production.py --timesteps 500000 --parallel-envs 8 --checkpoint-freq 25000
```

## Workspace Isolation

Each training run automatically creates a unique workspace with format:
`rl_production_{YYYYMMDD_HHMMSS}_{PROCESS_ID}`

**Examples:**
- `rl_production_20250919_143022_12345`
- `rl_production_20250919_143115_12347`

This ensures **no conflicts** when running multiple training sessions simultaneously.

## Parallel Training Examples

### Run Multiple Training Sessions Simultaneously
```bash
# Terminal 1: Small grid training
python scripts/train_rl_production.py --timesteps 200000 --parallel-envs 4 --workspace-base small_grid

# Terminal 2: Medium grid training
python scripts/train_rl_production.py --timesteps 500000 --parallel-envs 6 --workspace-base medium_grid

# Terminal 3: Large grid training
python scripts/train_rl_production.py --timesteps 1000000 --parallel-envs 8 --workspace-base large_grid
```

### Background Training (Linux/Mac)
```bash
# Run in background with nohup
nohup python scripts/train_rl_production.py --timesteps 500000 --parallel-envs 6 > training.log 2>&1 &

# Check progress
tail -f training.log

# Check running jobs
ps aux | grep train_rl_production
```

## macOS Parallel Training Note

**Important**: On macOS systems, parallel training may encounter numpy/multiprocessing compatibility issues. If you see errors like "Could not parse python long as longdouble", use single environment mode:

```bash
# Use single environment on macOS for compatibility
python scripts/train_rl_production.py --single-env --timesteps 100000

# Or add --single-env to any command for reliable macOS execution
python scripts/train_rl_production.py --timesteps 500000 --single-env --workspace-base macos_training
```

Single environment training is fully functional and provides the same training quality, just without the speed benefit of parallel environments.

## Monitoring Training Progress

### Real-time Log Monitoring
```bash
# Monitor specific training session
tail -f logs/rl_training_rl_production_20250919_143022_12345.log

# Monitor all training logs
tail -f logs/rl_training_*.log
```

### Check Training Status
```bash
# List all training sessions
ls -la logs/rl_training_*.log

# Check latest training summary
ls -la logs/training_summary_*.txt
cat logs/training_summary_$(ls -t logs/training_summary_*.txt | head -1)
```

### Monitor System Resources
```bash
# Monitor CPU and memory usage
htop

# Monitor GPU usage (if available)
nvidia-smi -l 1
```

## Training Configuration Options

### Scale Options
```bash
# Small scale (testing)
--timesteps 50000 --parallel-envs 2

# Medium scale (development)
--timesteps 200000 --parallel-envs 4

# Production scale (research)
--timesteps 500000 --parallel-envs 8

# Large scale (publication)
--timesteps 1000000 --parallel-envs 16
```

### Environment Configuration
```bash
# Single environment (debugging)
--single-env

# Custom parallel environments
--parallel-envs 6

# Custom workspace base name
--workspace-base experiment_1
```

### Model Management
```bash
# Custom model name
--model-name traffic_rl_v2

# Custom models directory
--models-dir experiments/models

# Checkpoint frequency
--checkpoint-freq 25000
```

### Logging Options
```bash
# Minimal output
--quiet

# Debug logging
--log-level DEBUG

# Custom log file
--log-file my_training.log

# Enable TensorBoard
--tensorboard
```

## Complete Example Commands

### Research-Grade Training Session
```bash
python scripts/train_rl_production.py \
  --timesteps 1000000 \
  --parallel-envs 8 \
  --checkpoint-freq 50000 \
  --model-name traffic_rl_research \
  --workspace-base research_run \
  --log-level INFO \
  --tensorboard
```

### Development Training Session
```bash
python scripts/train_rl_production.py \
  --timesteps 100000 \
  --parallel-envs 4 \
  --checkpoint-freq 10000 \
  --model-name traffic_rl_dev \
  --workspace-base dev_test \
  --log-level DEBUG
```

### Quick Validation Session
```bash
python scripts/train_rl_production.py \
  --timesteps 20000 \
  --single-env \
  --model-name traffic_rl_quick \
  --workspace-base validation \
  --quiet
```

## Expected Training Times

**Approximate training times** (will vary based on system):

| Timesteps | Parallel Envs | Estimated Time | Use Case |
|-----------|----------------|----------------|----------|
| 20,000    | 1              | 5-10 minutes   | Quick test |
| 100,000   | 4              | 20-30 minutes  | Development |
| 500,000   | 8              | 1-2 hours      | Research |
| 1,000,000 | 16             | 2-4 hours      | Publication |

## File Outputs

Each training session creates:

```
models/
├── traffic_rl_production_YYYYMMDD_HHMMSS.zip     # Trained model
├── traffic_rl_production_YYYYMMDD_HHMMSS.json    # Model metadata
└── best_model/                                   # Best model during training

logs/
├── rl_training_rl_production_YYYYMMDD_HHMMSS_PID.log  # Full training log
└── training_summary_rl_production_YYYYMMDD_HHMMSS_PID.txt  # Training summary

tensorboard_logs/  (if --tensorboard enabled)
└── rl_training_YYYYMMDD_HHMMSS/               # TensorBoard logs
```

## Troubleshooting

### Common Issues

**SUMO Connection Errors ("tcpip::Socket::recvAndCheck @ recv: peer shutdown"):**
This is common during long training runs (>50k timesteps). Solutions:
```bash
# Use more frequent checkpointing for recovery
--checkpoint-freq 10000

# Use shorter training runs and chain them together
python scripts/train_rl_production.py --timesteps 25000 --model-name run1
python scripts/train_rl_production.py --timesteps 25000 --resume-from models/run1_*.zip --model-name run2

# Use single environment for better stability
--single-env
```

**Out of Memory:**
```bash
# Reduce parallel environments
--parallel-envs 2

# Use single environment
--single-env
```

**Disk Space:**
```bash
# Check disk usage
df -h

# Clean old models
rm models/traffic_rl_production_*.zip
```

**Permission Issues:**
```bash
# Make script executable
chmod +x scripts/train_rl_production.py

# Check directory permissions
ls -la models/ logs/
```

### Getting Help
```bash
# Show all available options
python scripts/train_rl_production.py --help

# Validate environment
python scripts/train_rl_production.py --timesteps 1000 --single-env --log-level DEBUG
```

## Advanced Usage

### Custom Training Loop
```bash
# Run sequence of experiments
for envs in 2 4 8; do
  python scripts/train_rl_production.py \
    --timesteps 100000 \
    --parallel-envs $envs \
    --workspace-base "parallel_${envs}" \
    --model-name "traffic_rl_${envs}env"
done
```

### Experiment Tracking
```bash
# Create experiment directory
mkdir experiments/$(date +%Y%m%d)

# Run experiment with organized outputs
python scripts/train_rl_production.py \
  --timesteps 500000 \
  --parallel-envs 8 \
  --models-dir experiments/$(date +%Y%m%d)/models \
  --log-file experiments/$(date +%Y%m%d)/training.log
```

---

## Summary

**For immediate use, run:**
```bash
cd /Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator
python scripts/train_rl_production.py --timesteps 100000 --parallel-envs 4
```

This will start a production-scale training session with automatic workspace isolation, comprehensive logging, and proper parallel environment management.