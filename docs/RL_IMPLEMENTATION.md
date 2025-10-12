# RL Traffic Control Implementation Plan

This document provides a detailed implementation plan for adding reinforcement learning-based traffic signal control to the SUMO traffic generator. The implementation follows the design principles outlined in `RL_DISCUSSION.md`.

## Architecture Overview

The RL implementation extends the existing infrastructure by:

- Wrapping SUMO simulation with an OpenAI Gymnasium environment
- Reusing Tree Method's sophisticated traffic analysis capabilities
- Integrating with the existing TrafficController architecture
- Providing direct performance comparison with Tree Method baseline

### **CRITICAL: Decision Timing Alignment for Imitation Learning**

**Updated 2025-10-11**: For behavioral cloning / imitation learning to work correctly, the RL agent **MUST** make decisions at the **same frequency** as the expert (Tree Method):

- **Tree Method Decision Interval**: 90 seconds (`--tree-method-interval 90`)
- **RL Decision Interval**: 90 seconds (`DECISION_INTERVAL_SECONDS = 90`)
- **Demonstration Collection Interval**: 90 seconds (must match both)

**Why This Matters**:
- Behavioral cloning learns to imitate expert decisions **in specific contexts**
- If RL makes decisions 9x more frequently (10s vs 90s), it sees completely different traffic states
- The trained model achieves 100% accuracy on demonstrations but fails catastrophically in practice
- Completion rates: ~30% (10s interval) vs ~94% (Tree Method baseline)

**Key Insight**: The observation space at 10-second intervals is fundamentally different from 90-second intervals. A model trained on 90-second observations cannot meaningfully operate at 10-second intervals.

## Training Configuration Strategy

**Network-Specific Training Approach** (as recommended in RL_DISCUSSION.md):

- **Target Network**: 3�3 grid (9 intersections) for initial proof-of-concept
- **Vehicle Count**: 100-200 vehicles for balanced complexity
- **Simulation Time**: 30-60 minutes for manageable episode length
- **Training Scope**: Single fixed topology (faster training, optimal for research validation)

## Implementation Phases

### Phase 1: Dependencies and Structure  COMPLETED

**Duration**: 30 minutes
**Status**:  Done

#### Deliverables:

1. **RL Dependencies Added to requirements.txt**:

   - `gymnasium>=0.29.0` - Modern OpenAI Gym replacement
   - `stable-baselines3>=2.0.0` - PPO implementation with traffic-optimized hyperparameters
   - `torch>=2.0.0` - Neural network backend for policy learning

2. **Directory Structure Created**:
   - `src/rl/__init__.py` - Module initialization and documentation
   - `src/rl/constants.py` - RL-specific constants (no hardcoded values)
   - `src/rl/environment.py` - TrafficControlEnv class (Gymnasium interface)
   - `src/rl/vehicle_tracker.py` - VehicleTracker class (reward computation)
   - `src/rl/controller.py` - RLController class (traffic controller integration)
   - `src/rl/training.py` - PPO training pipeline

#### Key Features:

- **Comprehensive constants file** eliminates hardcoded values
- **Professional architecture** with proper abstractions and interfaces
- **Integration hooks** prepared for existing infrastructure
- **All dependencies verified** and working in virtual environment

---

### Phase 1.5: Configuration Decision (CRITICAL)

**Duration**: 15 minutes
**Status**: ✅ COMPLETED

#### Objectives:

Make concrete decisions about the training configuration that will determine state/action space dimensions and implementation requirements.

#### Critical Decisions Required:

**Network Configuration:**

- **Grid Dimension**: 3×3 (9 intersections) - FIXED DECISION
  - Rationale: Manageable complexity, 18 state features for signals, fast training
  - Alternative considered: 5×5 (too complex for initial proof-of-concept)
- **Block Size**: 150 meters - FIXED DECISION
  - Rationale: Realistic urban block size, tested in existing scenarios
- **Junctions to Remove**: 0 - FIXED DECISION
  - Rationale: Keep full 3×3 topology for coordination learning

**Traffic Configuration:**

- **Vehicle Count**: 150 vehicles - FIXED DECISION
  - Rationale: Sufficient for congestion without overwhelming small network
  - State space impact: Manageable vehicle tracking load
- **Simulation Duration**: 3600 seconds (1 hour) - FIXED DECISION
  - Rationale: Complete traffic cycles, reasonable episode length
- **Vehicle Types**: "passenger 90 public 10" - FIXED DECISION
  - Rationale: Realistic mix, consistent with existing scenarios

**RL-Specific Configuration:**

- **Decision Interval**: 10 seconds - FIXED DECISION
  - Rationale: Balance responsiveness with stability (from RL_DISCUSSION.md)
- **Measurement Interval**: 10 simulation steps - FIXED DECISION
  - Rationale: Frequent enough for reward computation, not overwhelming

#### Implementation Requirements:

**Configuration Class Creation:**

```python
@dataclass
class RLTrainingConfig:
    # Network parameters (FIXED for network-specific training)
    grid_dimension: int = 3
    block_size_m: int = 150
    junctions_to_remove: int = 0

    # Traffic parameters
    num_vehicles: int = 150
    vehicle_types: str = "passenger 90 public 10"
    end_time: int = 3600

    # RL-specific parameters
    decision_interval_seconds: int = 10
    measurement_interval_steps: int = 10

    # Derived properties
    @property
    def num_intersections(self) -> int:
        return self.grid_dimension * self.grid_dimension

    @property
    def state_vector_size(self) -> int:
        # E×4 + J×2 (edges×traffic_features + junctions×signal_features)
        # For 3×3 grid: ~12 edges × 4 + 9 junctions × 2 = 66 dimensions
        return self.estimated_edges * 4 + self.num_intersections * 2
```

**Validation Requirements:**

- Verify configuration works with existing SUMO pipeline
- Confirm state vector dimensions are reasonable for neural networks
- Test that vehicle count produces meaningful congestion patterns
- Validate simulation duration allows multiple traffic cycles

#### Rationale for Fixed Values:

**Why 3×3 Grid:**

- **State Space**: ~66 dimensions (manageable for PPO)
- **Action Space**: 18 actions (9 intersections × 2 decisions)
- **Coordination Complexity**: Sufficient for learning but not overwhelming
- **Training Speed**: Fast episodes enable rapid iteration

**Why 150 Vehicles:**

- **Congestion Balance**: Creates meaningful traffic without gridlock
- **Vehicle Tracking**: Manageable load for reward computation
- **Episode Variety**: Different patterns across episodes
- **Realistic Density**: ~16-17 vehicles per intersection (reasonable urban density)

**Why 1-Hour Episodes:**

- **Traffic Cycles**: Multiple complete signal cycles for learning
- **Training Efficiency**: Not too long for PPO batch processing
- **Evaluation Consistency**: Standard duration for performance comparison
- **Realistic Time Scale**: Matches traffic engineering analysis periods

#### Phase 1.5 Deliverables:

- **RLTrainingConfig** class with fixed parameters
- **Configuration validation** confirming compatibility with existing infrastructure
- **State/action space dimensions** calculated and documented
- **Training configuration locked** for consistent Phase 2+ implementation

#### Critical Note:

**These values are FIXED for the network-specific training approach.** Changing them after Phase 2 implementation begins will require significant code modifications to state/action spaces and neural network architectures.

---

### Phase 1.6: Parallel Execution Architecture

**Duration**: 45 minutes
**Status**: ✅ COMPLETED

#### Objectives:

Implement parallel execution architecture for efficient RL training with workspace isolation to prevent file conflicts between concurrent SUMO simulations.

#### Problem Statement:

RL training requires thousands of episodes to converge. While individual SUMO simulations execute quickly (seconds, not hours), sequential execution of episodes would still be inefficient. Parallel execution with proper isolation enables:

- Multiple simultaneous SUMO simulations
- Reduced total training time through parallelization
- Isolated workspace directories preventing file conflicts
- Diverse traffic patterns through seed variation

#### 1.6.1 Enhanced Configuration Support (15 minutes)

**RLTrainingConfig Updates** (`src/rl/config.py`):

- Added `n_parallel_envs` parameter (default: 4 environments)
- Added `get_cli_args_for_env(env_index, base_workspace)` method for unique workspace generation
- Enhanced validation to ensure parallel execution parameters are within safe bounds
- Updated logging and summary output to include parallel execution information

**Constants Integration** (`src/rl/constants.py`):

```python
# Parallel Execution
DEFAULT_N_PARALLEL_ENVS = 1        # Number of parallel environments for training
MIN_PARALLEL_ENVS = 1              # Minimum parallel environments
MAX_PARALLEL_ENVS = 16             # Maximum parallel environments (memory constraint)
PARALLEL_WORKSPACE_PREFIX = "rl_training"  # Base name for parallel workspace directories
```

#### 1.6.2 Vectorized Environment Implementation (20 minutes)

**Training Pipeline Updates** (`src/rl/training.py`):

- **`create_vectorized_env()`**: Creates stable-baselines3 vectorized environments
- **`make_env()`**: Factory function for individual environment creation with unique workspace
- **Updated `train_rl_policy()`**: Supports both parallel and single environment training modes

**Vectorization Strategy**:

- **SubprocVecEnv**: Used for true parallel execution (n_envs > 1)
- **DummyVecEnv**: Used for single environment or debugging (n_envs = 1)
- **Workspace Isolation**: Each environment gets unique directory (`rl_training/env_000`, `env_001`, etc.)
- **Seed Diversity**: Each environment uses `base_seed + env_index` for varied traffic patterns

#### 1.6.3 Validation and Testing (10 minutes)

**Enhanced Validation** (`src/rl/validate_config.py`):

- Added parallel execution validation section
- Tests workspace generation for all configured environments
- Validates unique workspace paths and seed assignment
- Comprehensive logging of parallel configuration details

**Validation Features**:

- Ensures each environment gets unique workspace path
- Verifies seed variation across environments
- Tests CLI argument generation for all parallel environments
- Reports parallel execution configuration in detailed output

#### Phase 1.6 Deliverables:

✅ **Enhanced RLTrainingConfig** with parallel execution support and workspace isolation
✅ **Vectorized training pipeline** using stable-baselines3's parallel environment support
✅ **Comprehensive validation** ensuring parallel execution works correctly
✅ **Workspace isolation system** preventing file conflicts between concurrent simulations
✅ **Seed diversification** ensuring varied traffic patterns across environments

#### Integration Results:

**Configuration Validation**:

```
PARALLEL EXECUTION:
  Parallel Environments: 4
  Workspace Isolation: Enabled (unique workspace per environment)
  Seed Variation: Base seed + environment index for diversity
```

**Workspace Generation**:

```
Environment 0: workspace=rl_training/env_000, seed=42
Environment 1: workspace=rl_training/env_001, seed=43
Environment 2: workspace=rl_training/env_002, seed=44
Environment 3: workspace=rl_training/env_003, seed=45
```

#### Performance Impact:

- **Training Acceleration**: 4x speedup with 4 parallel environments (theoretical maximum)
- **Memory Scaling**: Linear memory usage per environment (monitored by MAX_PARALLEL_ENVS)
- **Disk Isolation**: Each environment generates files in separate workspace directories
- **Reproducibility**: Fixed seed variation ensures consistent experimental results

#### Future Considerations:

- **Scaling**: Can increase parallel environments up to MAX_PARALLEL_ENVS (16) based on system resources
- **Distributed Training**: Architecture prepared for potential distributed training across multiple machines
- **Resource Monitoring**: Framework ready for resource usage monitoring and optimization

---

### Phase 2: Core RL Environment

**Duration**: 2-3 hours
**Status**: = Next Phase

#### Objectives:

Implement the core Gymnasium environment that wraps SUMO simulation and provides the RL interface.

#### 2.1 TrafficControlEnv Implementation (1.5 hours)

**State Space Design**:

- **Macroscopic Representation**: Each edge contributes 4 normalized indicators:
  - Speed ratio (current_speed / speed_limit)
  - Density utilization (current_density / max_capacity)
  - Flow rate (vehicles_per_second / max_throughput)
  - Congestion flag (binary indicator for bottlenecks)
- **Signal State**: Each intersection contributes 2 indicators:
  - Current phase (normalized by total phases)
  - Remaining duration (normalized by max_phase_time)
- **Fixed Dimensions**: For network with E edges and J junctions: `E � 4 + J � 2`
- **Normalization**: All values in [0.0, 1.0] range for neural network stability

**Action Space Design**:

- **Phase + Duration Control**: Each intersection requires 2 discrete decisions
- **Discrete Action Space**: Simplified for stable learning
- **Phase Selection**: Choose from available signal phases (4-8 options typically)
- **Duration Selection**: Choose from predefined durations [10, 15, 20, 30, 45, 60, 90, 120] seconds
- **Safety Constraints**: Enforce minimum green times (5-10 seconds) and maximum durations

**Integration Points**:

- Reuse existing SUMO configuration and network files
- Leverage Tree Method's traffic flow calculations
- Maintain consistency with existing simulation parameters

#### 2.2 VehicleTracker Implementation (1 hour)

**Individual Vehicle Monitoring**:

- Track each vehicle journey from network entry to completion
- Record start times, routes, and accumulated waiting times
- Handle dynamic vehicle populations (vehicles entering/leaving during episodes)

**Reward Computation Strategy** (from RL_DISCUSSION.md):

- **Intermediate Vehicle Penalties**: �wait_time penalties at measurement intervals
- **Credit Assignment**: Time-windowed attribution to recent signal decisions
- **Episode Throughput Bonuses**: Completion rewards based on total finished vehicles
- **Dual Component Balance**: Tune � parameter for penalty/bonus weighting

**Data Structures**:

- `vehicle_histories`: Dictionary mapping vehicle_id � journey data
- `decision_timestamps`: Log of signal control decisions with timestamps
- Memory management for long episodes (sliding windows, cleanup intervals)

#### 2.3 State Collection Integration (30 minutes)

**Tree Method Integration**:

- Reuse sophisticated traffic flow analysis already implemented
- Avoid duplicating complex traffic calculations
- Ensure identical state representations for fair RL vs Tree Method comparison

**Normalization Pipeline**:

- Consistent [0.0, 1.0] scaling using network-specific parameters
- Handle edge cases (empty roads, maximum congestion scenarios)
- Validate state vector dimensions match neural network expectations

#### Phase 2 Deliverables:

- Fully functional TrafficControlEnv with proper Gymnasium interface
- Vehicle tracking system with dual reward components
- State collection pipeline integrated with existing infrastructure
- Comprehensive logging and debugging capabilities

---

### Phase 3: RL Controller Integration

**Duration**: 1 hour
**Status**: = Pending

#### Objectives:

Integrate RL controller into existing traffic control architecture for seamless operation.

#### 3.1 RLController Class Implementation (30 minutes)

**Architecture Integration**:

- Extend existing `TrafficController` abstract base class
- Follow same patterns as TreeMethodController, ActuatedController, FixedController
- Implement required methods: `initialize()`, `update()`, `cleanup()`

**Model Management**:

- Load trained PPO models using stable-baselines3
- Handle both training mode (no model) and inference mode (loaded model)
- Validate model compatibility with current network configuration
- Graceful error handling for missing or incompatible models

**Action Execution Pipeline**:

- Convert RL model outputs to SUMO TraCI commands
- Apply safety constraint validation (min/max green times)
- Execute coordinated signal changes across all intersections
- Synchronize with SUMO simulation timesteps

#### 3.2 Factory Integration (15 minutes)

**TrafficControllerFactory Updates**:

- Add 'rl' option to traffic control method selection
- Handle RL-specific initialization parameters (model paths, training mode)
- Maintain backward compatibility with existing methods

**Command Line Integration**:

- Support `--traffic_control rl` option
- Add optional `--rl_model_path` parameter for trained model loading
- Validate RL parameters and provide helpful error messages

#### 3.3 Performance Tracking (15 minutes)

**Statistics Integration**:

- Use same Graph object as other controllers for vehicle tracking
- Report RL-specific metrics (model inference time, action distribution)
- Maintain consistency with existing performance reporting format

#### Phase 3 Deliverables:

- RLController fully integrated into existing architecture
- Command-line support for RL traffic control
- Consistent performance reporting and logging

---

### Phase 4: Training Pipeline

**Duration**: 1-2 hours
**Status**: = Pending

#### Objectives:

Create complete training pipeline for learning traffic control policies using PPO.

#### 4.1 PPO Configuration (30 minutes)

**Hyperparameter Optimization** (based on RL_DISCUSSION.md recommendations):

- **Learning Rate**: 2e-4 (conservative for expensive simulation episodes)
- **Clip Range**: 0.1 (prevent large policy updates that destabilize learning)
- **Batch Size**: 1024 (balance stability with memory requirements)
- **Network Architecture**: MLP with traffic-appropriate hidden dimensions
- **Training Stability**: Conservative parameters due to simulation expense

**Traffic-Specific Adaptations**:

- Episode length matching simulation end-time
- Custom callbacks for traffic metrics monitoring
- Checkpoint frequency appropriate for long training runs
- GPU utilization for faster neural network training

#### 4.2 Training Orchestration (45 minutes)

**Training Loop Management**:

- Environment validation using stable-baselines3 built-in checks
- Progressive training with increasing difficulty (curriculum learning)
- Real-time monitoring of learning progress and policy performance
- Automatic model checkpointing and best-model preservation

**Debugging and Monitoring**:

- Traffic-specific metrics tracking (completion rates, average wait times)
- Learning curve visualization and convergence monitoring
- Policy behavior analysis (action distribution, exploration patterns)
- Integration with existing logging infrastructure

#### 4.3 Model Management (15-30 minutes)

**Model Persistence**:

- Structured model saving with version control and metadata
- Model loading verification and compatibility checking
- Support for model fine-tuning and transfer learning
- Documentation of training configuration and performance

#### Phase 4 Deliverables:

- Complete PPO training pipeline with traffic-optimized hyperparameters
- Training monitoring and debugging capabilities
- Model management system with versioning and validation

---

### Phase 4.5: First Training Run

**Duration**: 45 minutes - 1 hour
**Status**: ✅ COMPLETED

#### Objectives:

Bridge the gap between training pipeline implementation and testing phases by executing the first actual training run to generate real trained models and validate end-to-end functionality.

#### 4.5.1 Pre-Training Validation (10 minutes)

**Environment Setup Verification**:

- Verify all RL dependencies available (stable-baselines3, tensorboard)
- Validate training environment configuration (3×3 grid, 150 vehicles)
- Test single environment creation and Gymnasium compliance
- Verify workspace and model directories are accessible

**Quick Integration Test**:

- Test TrafficControlEnv creation without errors
- Verify state/action space dimensions match expectations
- Test single step execution (reset → step → close)
- Validate callback initialization and logging setup

#### 4.5.2 Short Training Execution (20-30 minutes)

**Training Configuration**:

- **Network**: 3×3 grid, 150 vehicles, 1-hour episodes (from fixed config)
- **Training Steps**: 20,000-50,000 timesteps (short but meaningful)
- **Parallel Envs**: 2-4 environments (based on system capabilities)
- **Callbacks**: TrafficMetricsCallback, CheckpointCallback, EvalCallback enabled

**Execution Process**:

- Initialize training using `src/rl/train_model.py` script
- Monitor progress via console output and TensorBoard
- Verify callbacks working (checkpoints, evaluation, metrics)
- Capture training logs and any errors/warnings
- Wait for completion and verify model artifacts created

#### 4.5.3 Post-Training Validation (10 minutes)

**Model Verification**:

- Verify trained model loads successfully
- Test model inference on sample observations
- Validate model metadata completeness
- Check TensorBoard logs are accessible

**Quick Performance Check**:

- Run brief evaluation (5-10 episodes) using trained model
- Compare RL agent actions vs random baseline
- Verify agent produces valid traffic signal commands
- Document any obvious behavioral patterns

#### Phase 4.5 Deliverables:

- First successfully trained RL model with metadata
- TensorBoard training logs and learning curves
- Training session documentation and results summary
- Validated end-to-end training pipeline functionality

---

### Phase 5: Testing and Validation

**Duration**: 1 hour
**Status**: = Pending

#### Objectives:

Validate all RL components work correctly and produce reasonable behaviors.

#### 5.1 Component Testing (30 minutes)

**Environment Validation**:

- Gymnasium environment compliance using `check_env()`
- State vector dimensionality and normalization verification
- Action space validation and constraint enforcement testing
- Reward computation accuracy with known traffic scenarios

**Integration Testing**:

- SUMO simulation lifecycle management (start, step, close)
- TraCI command execution and error handling
- Vehicle tracking accuracy across different traffic patterns

#### 5.2 Consistency Validation (20 minutes)

**Tree Method Comparison**:

- State vector consistency between RL environment and Tree Method
- Identical traffic flow measurements for same simulation conditions
- Performance metric consistency using same evaluation criteria

**Behavioral Testing**:

- RL actions produce valid signal timing sequences
- Safety constraints prevent unrealistic signal behaviors
- Agent exploration stays within acceptable bounds

#### 5.3 End-to-End Validation (10 minutes)

**Training Pipeline Testing**:

- Short training runs complete without errors
- Model saving and loading works correctly
- Policy improvement observable over training iterations

#### Phase 5 Deliverables:

- Comprehensive test suite for all RL components
- Validation reports confirming correct behavior
- Integration testing with existing infrastructure

---

### Phase 6: Evaluation Framework

**Duration**: 30 minutes
**Status**: = Pending

#### Objectives:

Create rigorous evaluation framework for comparing RL performance against baselines.

#### 6.1 Performance Comparison (20 minutes)

**RL vs Tree Method Evaluation**:

- Identical network configurations and traffic scenarios
- Statistical significance testing with multiple independent runs
- Comprehensive metrics: throughput, waiting times, completion rates
- Fair comparison using same state representations and evaluation criteria

**Baseline Comparisons**:

- RL vs Tree Method (primary comparison)
- RL vs SUMO Actuated (secondary baseline)
- RL vs Fixed Timing (basic baseline)

#### 6.2 Statistical Analysis (10 minutes)

**Evaluation Methodology**:

- Minimum 20 independent runs per configuration
- 95% confidence intervals for performance metrics
- Effect size calculation and practical significance assessment
- Success criteria based on RL_DISCUSSION.md recommendations

#### Phase 6 Deliverables:

- Automated evaluation scripts for RL vs baseline comparison
- Statistical analysis tools for performance assessment
- Comprehensive evaluation reports with confidence intervals

---

## Success Criteria

**Technical Milestones**:

- RL environment passes Gymnasium validation
- Training converges to stable policies
- Learned policies outperform random actions
- Integration maintains existing system functionality

**Performance Targets** (based on RL_DISCUSSION.md):

- **Throughput**: 5-15% improvement vs Tree Method
- **Waiting Time**: 10-25% reduction vs baselines
- **Service Quality**: Better 90th percentile performance
- **Consistency**: Stable performance across different seeds

**Research Validation**:

- Fair comparison using identical simulation conditions
- Statistical significance of performance improvements
- Reproducible results with documented configuration
- Foundation for scaling to larger networks

## Implementation Notes

**Development Strategy**:

- Build incrementally with testing at each phase
- Maintain backward compatibility with existing functionality
- Use constants and configuration files (no hardcoded values)
- Follow existing code patterns and architectural conventions

**Quality Assurance**:

- Comprehensive logging and error handling
- Professional software engineering practices
- Integration with existing validation and testing frameworks
- Documentation and code comments for maintainability

**Future Extensions**:

- Network-agnostic training for multiple topologies
- Advanced reward shaping and curriculum learning
- Distributed training for larger networks
- Real-world deployment considerations

## Implementation Status Update

### Completed Phases:

✅ **Phase 1: Dependencies and Structure** - All RL dependencies and module structure implemented
✅ **Phase 1.5: Configuration Decision** - Fixed training configuration with immutable parameters established
✅ **Phase 1.6: Parallel Execution Architecture** - Vectorized environments with workspace isolation implemented
✅ **Phase 2: Core RL Environment** - TrafficControlEnv with Gymnasium interface and SUMO integration implemented
✅ **Phase 3: RL Controller Integration** - RLController integrated into existing traffic control architecture
✅ **Phase 4: Training Pipeline** - Complete PPO training pipeline with traffic-optimized hyperparameters implemented
✅ **Phase 4.5: First Training Run** - Production-scale training successfully executed with 50k+ timesteps

### RL System Status: **FULLY OPERATIONAL**

**All Core Components Implemented:**

- ✅ **TrafficControlEnv**: Complete Gymnasium environment with SUMO integration
- ✅ **VehicleTracker**: Journey tracking and dual-component reward computation
- ✅ **RLController**: Model loading, action execution, and traffic controller integration
- ✅ **Training Pipeline**: PPO training with parallel environments and workspace isolation
- ✅ **Production Scripts**: Independent training execution (`scripts/train_rl_production.py`)
- ✅ **Model Resume**: Chained training functionality for extending existing models

### Production Training Infrastructure:

**Independent Execution Commands:**

```bash
# Quick training (50k timesteps)
python scripts/train_rl_production.py --timesteps 50000 --parallel-envs 4

# Production training (100k+ timesteps)
python scripts/train_rl_production.py --timesteps 500000 --parallel-envs 8

# Chain training from existing model
python scripts/train_rl_production.py --timesteps 100000 --resume-from models/traffic_rl_*.zip

# Single environment (debugging)
python scripts/train_rl_production.py --timesteps 20000 --single-env
```

**Key Features:**

- **Workspace Isolation**: Unique timestamped workspaces prevent parallel training conflicts
- **Error Recovery**: Automatic fallback to single environment if parallel training fails
- **Comprehensive Logging**: TensorBoard integration, training summaries, progress monitoring
- **Model Management**: Automatic checkpointing, resume functionality, metadata preservation

### Validation Results:

**Training Success:**

- ✅ 50k timestep training completed successfully (102% completion at 51,200/50,000 timesteps)
- ✅ Proper learning curves observed with convergence patterns
- ✅ Model artifacts created and loadable (PPO model + metadata)
- ✅ Parallel environment training functional with workspace isolation

**Technical Validation:**

- ✅ Gymnasium environment compliance (`check_env()` passed)
- ✅ State/action space dimensions correct (~66 state, 18 actions for 3×3 grid)
- ✅ SUMO integration stable with TraCI command execution
- ✅ Reward computation functional with vehicle journey tracking

**Integration Validation:**

- ✅ RL controller accessible via `--traffic_control rl`
- ✅ Consistent with existing traffic control architecture
- ✅ Performance metrics compatibility with other controllers

### Current Implementation Status:

**✅ FULLY IMPLEMENTED (Phases 1-4.5)**:

- Complete end-to-end RL training and inference system
- Production-scale parallel training with workspace isolation
- Independent execution scripts for token-efficient training
- Model resume functionality for chained training workflows

**📋 NEXT PHASE (Phase 5: Testing and Validation)**:

- Component testing and comprehensive validation
- Automated test scripts for RL components
- Baseline performance comparison (RL vs Tree Method vs Actuated vs Fixed)
- Statistical evaluation framework for research validation

### Ready for Production Use:

The RL system is **fully operational** and ready for:

1. **Production Training**: Multi-hour, high-timestep training runs
2. **Research Evaluation**: Systematic comparison against baseline methods
3. **Performance Analysis**: Traffic signal optimization assessment
4. **Independent Operation**: Token-efficient training without Claude assistance

# ======================================================

# ======================================================

ARiel: From what I understand PPO takes care of how 'old' descisions infloence the current behaviour and we don't need to take care of it

## models:

Reward Function Summary

Overview

The reward function is a multi-objective optimization function that guides the reinforcement learning agent to optimize
traffic signal control across four key dimensions: throughput, waiting time minimization, speed maintenance, and bottleneck
reduction.

Design Philosophy

The reward function balances multiple competing objectives rather than optimizing a single metric. This approach ensures the
agent learns to manage trade-offs inherent in traffic control, such as:

- Maximizing vehicle completions while minimizing overall network waiting time
- Maintaining good flow speeds while preventing localized congestion
- Clearing insertion queues while avoiding bottleneck formation

---

Reward Components and Coefficients

1. Throughput Reward (Primary Goal)

Coefficient: 50.0 per vehicle

What it measures: Number of vehicles that successfully complete their journey during each timestep.

Purpose: Provides the strongest positive signal to encourage the network to move vehicles from origin to destination
efficiently.

Contribution to total reward: ~9-12% of average magnitude

- In low traffic: ~11.5%
- In high traffic: ~8.7%

Why this coefficient: Increased from initial 10.0 to 50.0 (5x) to strengthen the positive learning signal and counterbalance
penalty components.

---

2. Waiting Time Penalty (Secondary Goal)

Coefficient: 0.5 per vehicle-second

What it measures: Total accumulated waiting time across all vehicles currently in the network.

Purpose: Penalizes the agent when vehicles spend excessive time stopped or moving very slowly, encouraging continuous flow.

Contribution to total reward: ~53-58% of average magnitude

- In low traffic: ~53.5%
- In high traffic: ~58.0%

Why this coefficient: Reduced from initial 2.0 to 0.5 (4x decrease) to prevent this component from overwhelming all other
signals. Even at 0.5, it remains the dominant component but allows other objectives to influence learning.

Design constraint: Kept below 60% dominance to ensure multi-objective learning.

---

3. Speed Reward (Tertiary Goal)

Coefficient: 10.0 × (average_speed / 50.0)

What it measures: Network-wide average vehicle speed, normalized by 50 km/h (typical urban speed limit).

Purpose: Rewards smooth, continuous traffic flow and penalizes stop-and-go conditions even when vehicles aren't technically
"waiting."

Contribution to total reward: ~14-19% of average magnitude

- In low traffic: ~14.2%
- In high traffic: ~1.9% (lower speeds during congestion)

Why this coefficient: Increased from initial 2.0 to 10.0 (5x) to make speed improvements visible to the learning algorithm.
The normalization by 50 km/h keeps rewards in a reasonable range.

---

4. Bottleneck Penalty (Tertiary Goal)

Coefficient: 4.0 per bottleneck edge

What it measures: Number of network edges experiencing severe congestion (defined as edges with density approaching maximum
capacity and very low speeds).

Purpose: Penalizes localized congestion hotspots that can cascade into gridlock. Encourages spatially distributed traffic
flow.

Contribution to total reward: ~17-31% of average magnitude

- In low traffic: ~16.6%
- In high traffic: ~30.8% (more bottlenecks during congestion)

Why this coefficient: Increased from initial 0.5 → 2.0 → 4.0 (8x total) to strengthen the correlation between reward and
bottleneck reduction. This component shows excellent sensitivity to congestion levels.

---

5. Excessive Waiting Penalty

Coefficient: 2.0 per vehicle waiting >5 minutes

What it measures: Count of individual vehicles stuck waiting beyond 300 seconds (5 minutes).

Purpose: Provides an additional penalty for extreme cases where vehicles become severely delayed, preventing scenarios where a
few vehicles are sacrificed for overall network flow.

Contribution to total reward: 0% in validated scenarios (no vehicles exceeded threshold)

Why this coefficient: Increased from 0.5 to 2.0 (4x) to ensure severe delays are heavily penalized when they occur. Not active
in validation runs due to acceptable traffic conditions.

Threshold: 300 seconds (5 minutes)

---

6. Insertion Bonus

Coefficient: 1.0 (fixed)

What it measures: Provides a small constant bonus at each timestep when the insertion queue is below threshold.

Purpose: Encourages the network to maintain capacity for new vehicles entering the system.

Contribution to total reward: ~4-5% of average magnitude

Threshold: Bonus applied when fewer than 50 vehicles are waiting to enter the network

Why this coefficient: Small fixed value to provide consistent positive signal without dominating other components.

---

Validation Results

Correlation Performance

The reward function has been validated to correctly identify good vs. bad traffic states:

Low Traffic Scenario (500 vehicles, 7200s):

- Reward ↔ Speed: +0.413 (positive correlation - higher speeds = higher rewards)
- Reward ↔ Bottlenecks: -0.430 (negative correlation - more bottlenecks = lower rewards)
- All component correlations: 1.000 (perfect - each component measures exactly what it should)

High Traffic Scenario (1500 vehicles, 3600s):

- Reward ↔ Speed: +0.684 (stronger in congestion)
- Reward ↔ Bottlenecks: -0.708 (stronger in congestion)
- All component correlations: 1.000 (perfect)

Component Balance

The reward function maintains proper balance across traffic densities:

Balance Target: No single component should contribute >60% of total reward magnitude

Achieved Balance:

- Low traffic: 53.5% waiting, 16.6% bottleneck, 14.2% speed, 11.5% throughput
- High traffic: 58.0% waiting, 30.8% bottleneck, 1.9% speed, 8.7% throughput

All scenarios remain within the <60% dominance threshold.

Generalization

Key finding: The reward function shows stronger learning signals under congestion:

- Correlations increase from ~0.42 to ~0.70 when traffic density rises
- This is ideal because the agent learns better from challenging scenarios
- Coefficients don't need adjustment for different traffic levels

---

Design Decisions and Rationale

Why Multi-Objective?

Traffic control requires balancing competing goals. A single-metric reward (e.g., only throughput) would create pathological
behaviors:

- Throughput-only: Agent might create gridlock by allowing too many vehicles simultaneously
- Waiting-only: Agent might starve some directions to minimize total waiting
- Speed-only: Agent might sacrifice throughput for smooth flow of few vehicles

Coefficient Tuning Process

The coefficients were validated through iterative testing:

1. Initial coefficients (all equal weight): Waiting penalty dominated at 99%
2. First rebalancing (reduce waiting 4x, increase others 4-5x): Improved to 56-67% waiting dominance
3. Second rebalancing (double bottleneck penalty): Achieved 53-58% waiting dominance
4. Validation across scenarios: Confirmed generalization to different traffic densities

Why Waiting Penalty Remains Dominant

Even at reduced weight (0.5), waiting time contributes 53-58% of total reward because:

- It measures cumulative network-wide state (all vehicles × their waiting times)
- Other components measure discrete events (completions) or instantaneous state (speed, bottlenecks)
- This natural dominance is acceptable as long as it stays below 60%
- Waiting time is indeed the most important metric for traffic management

---

Mathematical Formulation

At each timestep, the total reward is computed as:

total_reward = + (throughput_reward) + (waiting_penalty)

- (excessive_waiting_penalty) + (speed_reward) + (bottleneck_penalty) + (insertion_bonus)

Where:

- throughput_reward = 50.0 × vehicles_completed_this_step
- waiting_penalty = -0.5 × Σ(waiting_time for each vehicle)
- excessive_waiting_penalty = -2.0 × count(vehicles with waiting_time > 300s)
- speed_reward = 10.0 × (network_average_speed / 50.0)
- bottleneck_penalty = -4.0 × count(bottleneck_edges)
- insertion_bonus = 1.0 (if vehicles_waiting_to_insert < 50, else 0.0)

---

### grid_dimension 3 block_size_m 150 num_vehicles 150 end-time 3600 seed 42:

- Original model: rl_traffic_production_20250928_151124_20250928_151319.zip
- Best checkpoint: rl_traffic_model_400000_steps.zip
- best_model name: dimension_3_block_150_vehicles_150_end_3600_seed_42.zip
- Train original: python scripts/train_rl_production.py --timesteps 1000000 --single-env
- Train resume from checkpoint: python scripts/train_rl_production.py --timesteps 1000000 --single-env --resume-from models/checkpoint/rl_traffic_model_410000_steps.zip
- Execution: env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control rl --rl_model_path models/best_model/dimension_3_block_150_vehicles_150_end_3600_seed_42.zip --grid_dimension 3 --block_size_m 150 --num_vehicles 150 --end-time 3600 --seed 42
- Compare: env PYTHONUNBUFFERED=1 python -m src.cli --traffic_control tree_method --grid_dimension 3 --block_size_m 150 --num_vehicles 150 --end-time 3600 --seed 42

### grid_dimension 5 block_size_m 200 num_vehicles 4500 end-time 7200 seeds 42 418655 166903:

- Train original:
  env PYTHONHASHSEED=0 NPY_DISABLE_LONGDOUBLE_FPFFLAGS=1 python scripts/train_rl_production.py --timesteps 200000 --single-env --checkpoint-freq 10000 --env-params "--network-seed 42 --grid_dimension 5 --block_size_m 200 --lane_count realistic --step-length 1.0 --land_use_block_size_m 25.0 --attractiveness land_use --traffic_light_strategy opposites --num_vehicles 4500 --routing_strategy 'shortest 75 realtime 25' --vehicle_types 'passenger 95 public 5' --passenger-routes 'in 20 out 20 inner 10 pass 50' --public-routes 'in 0 out 0 inner 0 pass 100' --departure_pattern uniform --private-traffic-seed 418655 --public-traffic-seed 166903 --end-time 7200 --start_time_hour 8.0"

- Train resume from checkpoint:
  env PYTHONHASHSEED=0 NPY_DISABLE_LONGDOUBLE_FPFFLAGS=1 python scripts/train_rl_production.py --timesteps 120000 --single-env --checkpoint-freq 10000 --resume-from models/checkpoint/rl_traffic_model_80000_stepszip.zip --env-params "--network-seed 42 --grid_dimension 5 --block_size_m 200 --lane_count realistic --step-length 1.0 --land_use_block_size_m 25.0 --attractiveness land_use --traffic_light_strategy opposites --num_vehicles 4500 --routing_strategy 'shortest 75 realtime 25' --vehicle_types 'passenger 95 public 5' --passenger-routes 'in 20 out 20 inner 10 pass 50' --public-routes 'in 0 out 0 inner 0 pass 100' --departure_pattern uniform --private-traffic-seed 418655 --public-traffic-seed 166903 --end-time 7200 --start_time_hour 8.0"

- Execution:
  env PYTHONUNBUFFERED=1 python -m src.cli --network-seed 42 --grid_dimension 5 --block_size_m 200 --lane_count realistic --step-length 1.0 --land_use_block_size_m 25.0 --attractiveness land_use --traffic_light_strategy opposites --num_vehicles 4500 --routing_strategy 'shortest 75 realtime 25' --vehicle_types 'passenger 95 public 5' --passenger-routes 'in 20 out 20 inner 10 pass 50' --public-routes 'in 0 out 0 inner 0 pass 100' --departure_pattern uniform --private-traffic-seed 418655 --public-traffic-seed 166903 --end-time 7200 --start_time_hour 8.0 --traffic_control rl --rl_model_path models/checkpoint/rl_traffic_model_80000_stepszip.zip

  Implementation Priority (Recommended Order) │
  │ │
  │ 2. Phase 1.1: Rebalance rewards with progressive bonuses │
  │ 3. Phase 1.2: Fix credit assignment with longer window │

# ===================

Imitation Learning (Behavior Cloning) would be a powerful way to bootstrap your RL agent using
Tree Method's knowledge. Let me explain the approach:

What is Behavior Cloning for Your Use Case?

Concept: Use Tree Method as an "expert teacher" to pre-train the RL policy, then fine-tune with RL.

Why this helps:

- Tree Method already finds good signal coordination patterns
- RL agent starts from "good" instead of "random"
- Dramatically reduces the 4^25 action space exploration problem
- Likely to converge in 100k-200k steps instead of never

Implementation Approaches:

Option A: Pure Pre-training (Recommended for 5×5)

Step 1: Collect Expert Demonstrations

# Run Tree Method on diverse scenarios, record (state, action) pairs

demonstrations = []
for scenario in range(1000): # 1000 different traffic scenarios # Run Tree Method
states, actions = run_tree_method_and_record(scenario)
demonstrations.extend(zip(states, actions))

Step 2: Train RL Policy to Imitate

# Supervised learning: minimize loss between RL action and Tree Method action

for epoch in range(10):
for state, expert_action in demonstrations:
rl_action_probs = policy_network(state)
loss = cross_entropy(rl_action_probs, expert_action)
loss.backward()

Step 3: Fine-tune with RL

# Start RL training from pre-trained policy

model = PPO.load("pretrained_from_tree_method.zip")
model.learn(total_timesteps=100000) # Fine-tune with RL rewards

Option B: Mixed Training (DAgger-style)

Start with Tree Method, gradually shift to RL:
beta = 1.0 # Start with 100% Tree Method
for timestep in training:
if random() < beta:
action = tree_method_action(state) # Expert
else:
action = rl_policy_action(state) # Learner

      beta = beta * 0.9999  # Gradually reduce expert influence

Option C: Reward Shaping

Add bonus when RL matches Tree Method:
rl_action = model.predict(state)
tree_action = tree_method.decide(state)

reward = traffic_reward # Normal reward
if rl_action == tree_action:
reward += 10.0 # Bonus for matching expert

Recommended Approach for You:

Option A (Pure Pre-training) because:

1. Tree Method is deterministic - easy to collect demonstrations
2. 5×5 network is fixed - demonstrations are reusable
3. No need to modify training loop
4. Most likely to succeed given the 4^25 action space

Implementation Plan:

Would you like me to:

1. Create a script to collect Tree Method demonstrations?

   - Run Tree Method 1000 times with different traffic seeds
   - Record (state, action) pairs
   - Save to training dataset

2. Create a behavior cloning pre-training script?

   - Load demonstrations
   - Train RL policy network to imitate Tree Method
   - Save pre-trained model

3. Modify training to use pre-trained initialization?

   - Load pre-trained weights instead of random initialization
   - Fine-tune with RL

This approach could reduce your training time from "never converges" to "converges in 100k steps".

Which option interests you most?

Three-Phase Approach │
│ │
│ Phase 1: Demonstration Collection Script (scripts/collect_tree_method_demonstrations.py) │
│ - Run Tree Method on 500-1000 diverse traffic scenarios (different seeds) │
│ - At each decision step, record: (state, tree_method_action) │
│ - Save demonstrations to data/demonstrations/tree_method_demos.npz │
│ - Estimated collection time: 10-20 hours for 1000 episodes │
│ │
│ Phase 2: Behavioral Cloning Pre-training (scripts/pretrain_from_demonstrations.py) │
│ - Load demonstrations dataset │
│ - Extract RL policy network from PPO model │
│ - Train policy to minimize cross-entropy loss between RL actions and Tree Method actions │
│ - Use PyTorch/TensorFlow supervised learning (not RL) │
│ - Save pre-trained model to models/pretrained_from_tree_method.zip │
│ - Training time: 30-60 minutes │
│ │
│ Phase 3: RL Fine-tuning (modify scripts/train_rl_production.py) │
│ - Add --pretrain-from argument │
│ - Load pre-trained weights instead of random initialization │
│ - Continue with normal PPO training to optimize for actual rewards │
│ - Expected improvement: Converge in 100k-200k steps vs never converging from random │
│ │
│ Key Technical Challenges │
│ │
│ 1. Action Space Mismatch: Tree Method outputs phase indices, RL outputs phase probabilities - need │
│ conversion │
│ 2. State Synchronization: Ensure RL state and Tree Method see identical observations │
│ 3. Dataset Diversity: Need varied traffic patterns for generalization │
│ 4. Hyperparameter Tuning: Learning rate, batch size for supervised pre-training

Implement Imitation Learning: Tree Method Expert Pre-training │ │
│ │ │ │
│ │ Phase 1: Demonstration Collection Adapter (30-45 min) │ │
│ │ │ │
│ │ Create src/rl/demonstration_collector.py: │ │
│ │ - TreeMethodDemonstrationAdapter class to observe Tree Method decisions │ │
│ │ - Read-only access to current_phase_durations attribute │ │
│ │ - Extract phase choices and convert to RL action format │ │
│ │ - State synchronization verification utility │ │
│ │ │ │
│ │ Phase 2: Demonstration Collection Script (45-60 min) │ │
│ │ │ │
│ │ Create scripts/collect_tree_method_demonstrations.py: │ │
│ │ - Run Tree Method simulations with varying seeds (500-1000 scenarios) │ │
│ │ - Use adapter to extract (state, action) pairs at each decision step │ │
│ │ - Save demonstrations to data/demonstrations/tree_method_demos.npz │ │
│ │ - Progress tracking and estimated time: 10-20 hours for 1000 episodes │ │
│ │ │ │
│ │ Phase 3: Behavioral Cloning Pre-training (45-60 min) │ │
│ │ │ │
│ │ Create scripts/pretrain_from_demonstrations.py: │ │
│ │ - Load demonstration dataset │ │
│ │ - Extract PPO policy network │ │
│ │ - Supervised learning: minimize cross-entropy loss between RL predictions and Tree Method actions │ │
│ │ - Save pre-trained model to models/pretrained_from_tree_method.zip │ │
│ │ - Training time: 30-60 minutes │ │
│ │ │ │
│ │ Phase 4: Training Pipeline Integration (15-30 min) │ │
│ │ │ │
│ │ Modify scripts/train_rl_production.py: │ │
│ │ - Add --pretrain-from <path> argument │ │
│ │ - Load pre-trained weights before RL fine-tuning │ │
│ │ - Otherwise use random initialization (preserve existing behavior) │ │
│ │ │ │
│ │ Phase 5: Testing and Validation (30 min) │ │
│ │ │ │
│ │ - Verify Tree Method standalone operation unchanged │ │
│ │ - Test demonstration collection on small scenario │ │
│ │ - Validate pre-training reduces to reasonable loss │ │
│ │ - Test RL fine-tuning from pre-trained initialization │ │
│ │ │ │
│ │ Expected Outcome: RL agent starts from "good" Tree Method policies instead of random, dramatically │ │
│ │ reducing training time from "never converges" to "converges in 100k-200k steps" │ │
│ │ │ │
│ │ Safety: Zero modifications to Tree Method code - all changes are new files or RL-only additions │ │
│

## Imitation Learning System (IMPLEMENTED)

**Status**: ✅ COMPLETED

The imitation learning system allows using Tree Method as an "expert teacher" to pre-train the RL policy, dramatically accelerating convergence.

### Implementation Summary

**Three-Phase Process:**

1. **Demonstration Collection**: Observe Tree Method simulations and collect (state, action) pairs
2. **Behavioral Cloning**: Train RL policy to imitate Tree Method using supervised learning
3. **RL Fine-tuning**: Use pre-trained model as initialization for PPO training

**Key Files:**

- `src/rl/demonstration_collector.py` - Read-only adapter to observe Tree Method decisions
- `scripts/collect_tree_method_demonstrations.py` - Demonstration collection script
- `scripts/pretrain_from_demonstrations.py` - Behavioral cloning pre-training script
- `src/rl/training.py` - Extended with `pretrain_from_model` parameter
- `scripts/train_rl_production.py` - Extended with `--pretrain-from` argument

**Design Principles:**

- ✅ **Standalone & Optional**: Doesn't modify existing training workflow
- ✅ **Backward Compatible**: Old training method works unchanged
- ✅ **No Code Duplication**: Reuses existing RL and Tree Method components
- ✅ **No Hardcoding**: All constants in `src/rl/constants.py`
- ✅ **Zero Tree Method Changes**: Read-only observation, no modifications

### Usage

**Quick Start:**

```bash
# 1. Collect demonstrations (10 scenarios for testing)
python scripts/collect_tree_method_demonstrations.py --scenarios 500 --base-seed 42 --config configs/tree_method_demonstrations_1.json

# 2. Pre-train policy
python scripts/pretrain_from_demonstrations.py --input models/tree_method_demonstration/demo_20251010_094243.npz --output models/pretrained/test_pretrained_20_20251010_094243.zip --epochs 20

# 3. Fine-tune with RL
python scripts/train_rl_production.py --timesteps 10000 --single-env --pretrain-from models/pretrained/tree_method_pretrained.zip --env-params "..."
```

**Production Workflow:**

```bash
# 1. Collect large dataset (500 scenarios)
python scripts/collect_tree_method_demonstrations.py --scenarios 500 --grid-dimension 5 --num-vehicles 1000 --end-time 3600

# 2. Pre-train (full training)
python scripts/pretrain_from_demonstrations.py --input data/demonstrations/tree_method_demos.npz --epochs 10

# 3. Fine-tune with RL (full training)
python scripts/train_rl_production.py --timesteps 500000 --parallel-envs 8 --pretrain-from models/pretrained/tree_method_pretrained.zip --env-params "..."
```

**For detailed documentation, see**: [`docs/IMITATION_LEARNING_GUIDE.md`](IMITATION_LEARNING_GUIDE.md)

### Technical Architecture

**TreeMethodDemonstrationAdapter** (`src/rl/demonstration_collector.py`):

- Read-only observer of Tree Method's phase selection decisions
- Extracts active phase per junction and converts to RL action format
- Verifies state synchronization between RL and Tree Method
- Zero modifications to Tree Method code

**Behavioral Cloning Training** (`scripts/pretrain_from_demonstrations.py`):

- Loads (state, action) pairs from demonstrations file
- Trains PPO policy network using supervised learning (cross-entropy loss)
- Multi-junction discrete action space with per-junction loss calculation
- 90/10 train/validation split for monitoring overfitting
- Saves training history and metadata

**RL Integration** (`src/rl/training.py`):

- New `pretrain_from_model` parameter in `train_rl_policy()`
- Loads pre-trained weights when provided (mutually exclusive with `resume_from_model`)
- Policy then fine-tuned with PPO algorithm
- All existing features (checkpointing, evaluation, callbacks) work normally

### Expected Benefits

- **Faster Convergence**: Pre-trained policy starts from expert knowledge instead of random
- **Reduced Training Time**: From "never converges" to "converges in 100k-200k steps"
- **Better Final Performance**: Expert initialization guides exploration toward better solutions
- **Lower Computational Cost**: Fewer training steps required to reach target performance

### Configuration Constants

All imitation learning parameters in `src/rl/constants.py`:

```python
# Demonstration Collection
DEMONSTRATION_COLLECTION_DEFAULT_SCENARIOS = 500
DEMONSTRATION_DECISION_INTERVAL_SECONDS = 10

# Behavioral Cloning Pre-training
PRETRAINING_LEARNING_RATE = 1e-3
PRETRAINING_BATCH_SIZE = 64
PRETRAINING_EPOCHS = 10
PRETRAINING_VALIDATION_SPLIT = 0.1

# Pre-trained Model Storage
PRETRAINED_MODEL_DIR = "models/pretrained"
```

### Safety & Compatibility

**Tree Method Safety:**

- ✅ No modifications to Tree Method code
- ✅ Read-only observation through demonstration adapter
- ✅ Tree Method standalone operation completely unchanged
- ✅ No shared mutable state between RL and Tree Method

**Training Compatibility:**

- ✅ `--pretrain-from` and `--resume-from` are mutually exclusive
- ✅ Old training method (without `--pretrain-from`) works unchanged
- ✅ All existing RL features (parallel envs, checkpointing, evaluation) compatible
- ✅ Backward compatible with existing models and workflows

# 1. Collect 10 demos with default config (quick test)

python scripts/collect_tree_method_demonstrations.py --scenarios 10 --base-seed 42

# Output: models/tree_method_demonstration/demo_20250109_143022.npz

# 2. Pre-train (5 epochs for quick test)

python scripts/pretrain_from_demonstrations.py --input models/tree_method_demonstration/demo_20251010_084940.npz --output models/pretrained/test_pretrained_20251010_084940.zip --epochs 5

# 3. Fine-tune with RL (short run)

# Note: env-params should match the fixed params from config

python scripts/train_rl_production.py --timesteps 10000 --single-env --checkpoint-freq 10000 --pretrain-from models/pretrained/test_pretrained_20251010_084940.zip --env-params "--network-seed 42 --grid_dimension 5 --block_size_m 200 --lane_count realistic --step-length 1.0 --land_use_block_size_m 25.0 --attractiveness land_use --traffic_light_strategy opposites --routing_strategy 'shortest 75 realtime 25' --vehicle_types 'passenger 95 public 5' --passenger-routes 'in 20 out 20 inner 10 pass 50' --public-routes 'in 0 out 0 inner 0 pass 100' --departure_pattern uniform --private-traffic-seed 418655 --public-traffic-seed 166903 --start_time_hour 8.0 --num_vehicles 1000 --end-time 3600"
