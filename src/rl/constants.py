"""
RL-specific constants for traffic signal control.

Constants specific to reinforcement learning components including
training parameters, reward system configuration, and RL environment settings.
General simulation constants are in src/constants.py.
"""

# =============================================================================
# RL TRAINING CONSTANTS
# =============================================================================

# PPO Training Parameters (Optimized for Long-Horizon Traffic Control)
import torch.nn as nn
DEFAULT_LEARNING_RATE = 3e-4  # Higher initial (will be scheduled)
DEFAULT_CLIP_RANGE = 0.2      # More aggressive for traffic adaptation
DEFAULT_BATCH_SIZE = 2048     # Larger for stability with long episodes
DEFAULT_N_STEPS = 4096        # More experience for long-horizon effects
DEFAULT_N_EPOCHS = 15         # More optimization (expensive simulation data)
DEFAULT_GAMMA = 0.995         # Longer horizon for cascading traffic effects
DEFAULT_GAE_LAMBDA = 0.98     # Better advantage estimation for long episodes
MAX_GRAD_NORM = 0.5           # Gradient clipping for stability

# Learning Rate Schedule
LEARNING_RATE_SCHEDULE_ENABLED = True
LEARNING_RATE_SCHEDULE_TYPE = "exponential"  # "exponential" or "linear"
LEARNING_RATE_INITIAL = 3e-4
LEARNING_RATE_FINAL = 5e-6
LEARNING_RATE_DECAY_RATE = 0.99995  # For exponential: smooth decay over ~1M steps

# Entropy Coefficient Schedule (Exploration → Exploitation)
ENTROPY_COEF_SCHEDULE_ENABLED = True
ENTROPY_COEF_INITIAL = 0.02   # High exploration early
ENTROPY_COEF_FINAL = 0.001    # Low exploration late (exploit learned policy)
ENTROPY_COEF_DECAY_STEPS = 500000  # Decay over 500k steps

# Early Stopping (Prevent Performance Degradation)
EARLY_STOPPING_ENABLED = True
# Stop after 10 evals (~100k steps) without improvement
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 10.0  # Minimum improvement threshold
EARLY_STOPPING_VERBOSE = True

# Training Duration
DEFAULT_TOTAL_TIMESTEPS = 100000
DEFAULT_CHECKPOINT_FREQ = 10000

# Parallel Execution
DEFAULT_N_PARALLEL_ENVS = 1        # Number of parallel environments for training
MIN_PARALLEL_ENVS = 1              # Minimum parallel environments
# Maximum parallel environments (memory constraint)
MAX_PARALLEL_ENVS = 16
# Base name for parallel workspace directories
PARALLEL_WORKSPACE_PREFIX = "rl_training"
SINGLE_ENV_THRESHOLD = 1           # Threshold for using DummyVecEnv vs SubprocVecEnv

# Model Paths
DEFAULT_MODEL_SAVE_PATH = "models/rl_traffic_policy"
DEFAULT_MODELS_DIRECTORY = "models"

# Phase 4: Training Pipeline Constants
# Hidden layer dimensions for MLP
TRAINING_NETWORK_ARCHITECTURE = [256, 256]
# Device selection (auto, cpu, cuda)
TRAINING_DEVICE_AUTO = "auto"
TRAINING_TENSORBOARD_LOG_DIR = "tensorboard_logs"  # TensorBoard logging directory
TRAINING_BEST_MODEL_PREFIX = "best_model"         # Prefix for best model saves
TRAINING_CHECKPOINT_PREFIX = "checkpoint"         # Prefix for checkpoint saves
# Extension for model metadata files
TRAINING_MODEL_METADATA_EXTENSION = ".json"
# Episodes to evaluate at each checkpoint
TRAINING_EVAL_EPISODES_PER_CHECKPOINT = 5
# Episodes to wait before early stopping
TRAINING_PATIENCE_EPISODES = 20
# Minimum improvement to reset patience
TRAINING_MIN_IMPROVEMENT_THRESHOLD = 0.01

# =============================================================================
# RL ENVIRONMENT CONSTANTS
# =============================================================================

# State Space Configuration
STATE_NORMALIZATION_MIN = 0.0
STATE_NORMALIZATION_MAX = 1.0

# State Feature Normalization Thresholds
# Maximum expected density for normalization
MAX_DENSITY_VEHICLES_PER_METER = 0.2
# Maximum expected flow for normalization
MAX_FLOW_VEHICLES_PER_SECOND = 1.0
# Seconds - waiting time threshold for congestion flag
CONGESTION_WAITING_TIME_THRESHOLD = 30.0

# Action Space Configuration - Traffic Signal Phases
# Number of phases per intersection (0-3)
NUM_TRAFFIC_LIGHT_PHASES = 4
NUM_PHASE_DURATION_OPTIONS = 8          # Number of duration options (0-7)

# Traffic Flow State Features per Edge
EDGE_FEATURES_COUNT = 4  # speed, density, flow, congestion_flag
EDGE_SPEED_FEATURE_INDEX = 0
EDGE_DENSITY_FEATURE_INDEX = 1
EDGE_FLOW_FEATURE_INDEX = 2
EDGE_CONGESTION_FEATURE_INDEX = 3

# Signal State Features per Junction
JUNCTION_FEATURES_COUNT = 2  # phase, remaining_duration
JUNCTION_PHASE_FEATURE_INDEX = 0
JUNCTION_DURATION_FEATURE_INDEX = 1

# Action Space Configuration
# phase + duration (legacy - for backward compatibility)
ACTIONS_PER_INTERSECTION = 2
PHASE_ACTION_INDEX = 0
DURATION_ACTION_INDEX = 1

# Phase-Only Action Space Configuration (Tree Method Compatibility)
# Enable phase-only control (matching Tree Method)
RL_PHASE_ONLY_MODE = True
# Fixed duration in seconds (matches Tree Method MIN_PHASE_TIME = 10)
# Note: Tree Method uses variable durations but applies changes every MIN_PHASE_TIME
# RL uses this as the fixed duration for each phase decision
RL_FIXED_PHASE_DURATION = 10
# Only phase selection per intersection (phase index 0-3)
RL_ACTIONS_PER_INTERSECTION_PHASE_ONLY = 1

# Phase Duration Options (seconds)
PHASE_DURATION_OPTIONS = [10, 15, 20, 30, 45, 60, 90, 120]
MIN_PHASE_DURATION = 10
MAX_PHASE_DURATION = 120

# Traffic Signal Constraints (seconds)
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 120
YELLOW_CLEARANCE_TIME = 3

# Decision Making Frequency
# IMPORTANT: Must match Tree Method's decision interval (90s) for behavioral cloning
# to work correctly. The RL agent should make decisions at the same frequency as
# the expert (Tree Method) it's imitating.
DECISION_INTERVAL_SECONDS = 90  # Match Tree Method interval for imitation learning
MIN_DECISION_INTERVAL = 5
MAX_DECISION_INTERVAL = 120  # Increased to accommodate Tree Method interval

# =============================================================================
# REWARD SYSTEM CONSTANTS
# =============================================================================

# Reward Components (Tunable weights for reward function validation)
# Weight for individual vehicle penalties (reduced to balance with progressive bonuses)
VEHICLE_PENALTY_WEIGHT = 2.0
# Weight for episode completion bonuses (α parameter) - legacy
THROUGHPUT_BONUS_WEIGHT = 10.0

# NEW: Multi-Objective Reward Function Weights (REBALANCED based on validation)
# Primary goal: Maximize throughput (vehicles completing journeys)
# Reward per vehicle completion (was 10.0, increased 5x)
REWARD_THROUGHPUT_PER_VEHICLE = 50.0
# Secondary goal: Minimize waiting time
# Multiplier for waiting time penalties (was 2.0, decreased 4x to reduce dominance)
REWARD_WAITING_TIME_PENALTY_WEIGHT = 0.5
# Extra penalty per vehicle waiting >5min (was 0.5, increased 4x)
REWARD_EXCESSIVE_WAITING_PENALTY = 2.0
# Threshold in seconds (5 minutes)
REWARD_EXCESSIVE_WAITING_THRESHOLD = 300.0
# Tertiary goal: Maintain network flow
# Reward for good average speed (was 2.0, increased 5x to make visible)
REWARD_SPEED_REWARD_FACTOR = 10.0
# Normalize speed by this value (km/h)
REWARD_SPEED_NORMALIZATION = 50.0
# Penalty per bottleneck edge (was 0.5→2.0→4.0, increased to strengthen correlation)
REWARD_BOTTLENECK_PENALTY_PER_EDGE = 4.0
# Bonus: Clear insertion queue
# Bonus when waiting queue < threshold
REWARD_INSERTION_BONUS = 1.0
REWARD_INSERTION_THRESHOLD = 50               # Max waiting vehicles for bonus

# Progressive Bonus Configuration (Legacy - can be disabled)
# Enable progressive bonus system
PROGRESSIVE_BONUS_ENABLED = True
# Per vehicle completion (immediate feedback)
IMMEDIATE_THROUGHPUT_BONUS_WEIGHT = 2.0
# Base bonus for performance streaks
PERFORMANCE_STREAK_BASE_BONUS = 0.5
# Exponential scaling for streaks
PERFORMANCE_STREAK_MULTIPLIER = 1.2
# Max waiting time threshold for "good performance"
PERFORMANCE_STREAK_THRESHOLD = 5.0
# Bonus per km/h average speed increase
SPEED_IMPROVEMENT_BONUS_FACTOR = 1.0
# Bonus per reduced bottleneck
CONGESTION_REDUCTION_BONUS = 0.8
# Bonuses at 25%, 50%, 75%, 90% completion
MILESTONE_COMPLETION_BONUSES = [2.0, 5.0, 8.0, 12.0]

# Vehicle Tracking
MEASUREMENT_INTERVAL_STEPS = 10   # Steps between reward measurements
CREDIT_ASSIGNMENT_WINDOW_STEPS = 50  # Time window for crediting decisions

# Progressive Bonus Tracking
# Steps to track for performance streaks
PERFORMANCE_STREAK_WINDOW_SIZE = 20
# Steps to track for speed improvements
SPEED_HISTORY_WINDOW_SIZE = 30
# Steps to track for congestion changes
CONGESTION_HISTORY_WINDOW_SIZE = 20
# Completion percentage thresholds
MILESTONE_COMPLETION_THRESHOLDS = [0.25, 0.50, 0.75, 0.90]

# Penalty Calculation
WAITING_TIME_PENALTY_FACTOR = -0.1  # Penalty per second of increased waiting time
MIN_WAITING_TIME_THRESHOLD = 1.0    # Minimum waiting time to trigger penalty

# Episode Rewards
COMPLETION_BONUS_PER_VEHICLE = 1.0   # Bonus per completed vehicle

# =============================================================================
# RL EVALUATION CONSTANTS
# =============================================================================

# Performance Evaluation
DEFAULT_EVAL_EPISODES = 10
MIN_EVAL_EPISODES = 5
MAX_EVAL_EPISODES = 50

# Success Criteria (based on RL_DISCUSSION.md recommendations)
THROUGHPUT_IMPROVEMENT_TARGET_MIN = 0.05  # 5% minimum improvement vs baseline
THROUGHPUT_IMPROVEMENT_TARGET_MAX = 0.15  # 15% target improvement vs baseline
WAITING_TIME_REDUCTION_TARGET_MIN = 0.10  # 10% minimum reduction vs baseline
WAITING_TIME_REDUCTION_TARGET_MAX = 0.25  # 25% target reduction vs baseline

# Statistical Evaluation
MIN_STATISTICAL_RUNS = 20  # Minimum runs for statistical significance
CONFIDENCE_LEVEL = 0.95    # 95% confidence intervals

# =============================================================================
# RL ARCHITECTURE CONSTANTS
# =============================================================================

# Network Architecture (from RL_DISCUSSION.md recommendations)
# Training Scope: Network-specific
TRAINING_SCOPE_NETWORK_SPECIFIC = True

# Agent Architecture: Centralized
AGENT_ARCHITECTURE_CENTRALIZED = True

# Input Model: Macroscopic
INPUT_MODEL_MACROSCOPIC = True

# Time Resolution: Fixed intervals
TIME_RESOLUTION_FIXED_INTERVALS = True

# Exploration: Constrained
EXPLORATION_CONSTRAINED = True

# =============================================================================
# VEHICLE TRACKING CONSTANTS
# =============================================================================

# Journey Tracking
MAX_TRACKED_VEHICLES = 10000      # Maximum vehicles to track simultaneously
VEHICLE_HISTORY_RETENTION = 1000  # Maximum historical records per vehicle

# Decision Tracking
MAX_DECISION_HISTORY = 1000  # Maximum decision records to keep
DECISION_CLEANUP_INTERVAL = 100  # Steps between decision history cleanup

# Performance Metrics
STATISTICS_UPDATE_INTERVAL = 50  # Steps between statistics updates

# =============================================================================
# RL LOGGING AND MONITORING CONSTANTS
# =============================================================================

# Logging Configuration
RL_LOG_LEVEL = "INFO"
RL_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Progress Tracking
TRAINING_PROGRESS_LOG_INTERVAL = 1000  # Steps between progress logs
EVALUATION_PROGRESS_LOG_INTERVAL = 1   # Episodes between evaluation logs

# =============================================================================
# ERROR HANDLING CONSTANTS
# =============================================================================

# RL Error Codes
RL_ERROR_ENVIRONMENT_INIT = "RL_001"
RL_ERROR_MODEL_LOAD = "RL_002"
RL_ERROR_ACTION_EXECUTION = "RL_003"
RL_ERROR_STATE_COLLECTION = "RL_004"
RL_ERROR_REWARD_COMPUTATION = "RL_005"
RL_ERROR_TRAINING_FAILURE = "RL_006"
RL_ERROR_EVALUATION_FAILURE = "RL_007"
RL_ERROR_VEHICLE_TRACKING = "RL_008"

# Error Message Templates
RL_ERROR_MSG_TEMPLATE = "RL ERROR [{code}]: {operation} failed: {reason}"
RL_VALIDATION_ERROR_MSG = "RL VALIDATION ERROR [{code}]: {component} validation failed: {details}"
RL_MODEL_ERROR_MSG = "RL MODEL ERROR [{code}]: Model '{model_path}' error: {reason}"

# =============================================================================
# INTEGRATION CONSTANTS
# =============================================================================

# SUMO Integration
SUMO_STEP_SYNC_MODE = True  # Synchronize with SUMO simulation steps
SUMO_STATE_COLLECTION_TIMEOUT = 5.0  # Seconds to wait for state collection

# Tree Method Integration
REUSE_TREE_METHOD_CALCULATIONS = True  # Reuse existing traffic analysis
TREE_METHOD_COMPATIBILITY_MODE = True  # Maintain compatibility with Tree Method

# =============================================================================
# RL CONTROLLER INTEGRATION CONSTANTS
# =============================================================================

# Controller Identification
RL_CONTROLLER_NAME = "rl"                    # Traffic control method name
RL_CONTROLLER_DISPLAY_NAME = "RL Agent"      # Human-readable controller name
RL_CONTROLLER_DESCRIPTION = "Deep Reinforcement Learning traffic signal control using PPO"

# Model Management
# No model path by default (training mode)
DEFAULT_RL_MODEL_PATH = None
# Stable-baselines3 model file extension
RL_MODEL_EXTENSION = ".zip"
# Seconds to wait for model validation
RL_MODEL_VALIDATION_TIMEOUT = 10.0
# Number of model loading retry attempts
RL_MODEL_LOAD_RETRIES = 3

# Action Execution
RL_ACTION_EXECUTION_TIMEOUT = 2.0            # Seconds for action execution
# Enable safety constraint validation
RL_SAFETY_CHECK_ENABLED = True
# Enforce minimum green time constraints
RL_MIN_GREEN_TIME_ENFORCEMENT = True
# Enforce maximum duration constraints
RL_MAX_DURATION_ENFORCEMENT = True

# Performance Monitoring
# Track model inference timing (disabled for production training)
RL_INFERENCE_TIME_TRACKING = False
# Track action selection patterns (disabled for production training)
RL_ACTION_DISTRIBUTION_TRACKING = False
# Steps between statistics collection
RL_STATISTICS_COLLECTION_INTERVAL = 100
# Enable performance logging (disabled for production training)
RL_PERFORMANCE_LOGGING_ENABLED = False

# Training vs Inference Mode
RL_TRAINING_MODE = "training"                # Training mode identifier
RL_INFERENCE_MODE = "inference"              # Inference mode identifier
# Default mode when no model provided
RL_DEFAULT_MODE = RL_TRAINING_MODE

# Error Handling
RL_CONTROLLER_ERROR_PREFIX = "RL_CONTROLLER"  # Error message prefix
RL_MODEL_COMPATIBILITY_CHECK = True          # Validate model compatibility
# Whether to fall back to fixed timing on errors
RL_GRACEFUL_DEGRADATION = False

# =============================================================================
# RL TRAINING CONFIGURATION (FIXED VALUES FOR NETWORK-SPECIFIC TRAINING)
# =============================================================================

# NOTE: Network configuration parameters (grid_dimension, num_vehicles, etc.)
# are now passed via --env-params at runtime. No fixed values here.

# =============================================================================
# DEVELOPMENT AND DEBUGGING CONSTANTS
# =============================================================================

# Default Values for Initialization
DEFAULT_INITIAL_STEP = 0              # Initial simulation step
DEFAULT_INITIAL_TIME = 0              # Initial simulation time
DEFAULT_INITIAL_PENALTY = 0.0         # Initial penalty value
DEFAULT_FALLBACK_VALUE = 0.0          # Default fallback for missing data
DEFAULT_OBSERVATION_PADDING = 0.0     # Value for padding observation vectors
# Standard simulation step length (seconds)
DEFAULT_STEP_LENGTH = 1.0
DEFAULT_TRAINING_SEED = 42            # Fixed seed for reproducible training

# Statistical Calculation Constants
STD_CALCULATION_MIN_VALUES = 1       # Minimum values needed for std calculation
STD_CALCULATION_FALLBACK = 0.0       # Fallback value when std calculation fails

# Debug Modes
DEBUG_STATE_COLLECTION = False    # Debug state vector construction
DEBUG_REWARD_COMPUTATION = False  # Debug reward calculations
DEBUG_ACTION_EXECUTION = False    # Debug action processing
DEBUG_VEHICLE_TRACKING = False    # Debug vehicle journey tracking

# Testing Configuration
TEST_GRID_DIMENSION = 3           # Small grid for testing
TEST_NUM_VEHICLES = 50            # Small vehicle count for testing
TEST_END_TIME = 1800              # 30 minutes for testing
TEST_TRAINING_TIMESTEPS = 1000    # Short training for testing

# Validation Thresholds
MIN_STATE_VECTOR_SIZE = 10        # Minimum expected state vector size
MAX_STATE_VECTOR_SIZE = 1000      # Maximum reasonable state vector size
MIN_ACTION_VECTOR_SIZE = 2        # Minimum expected action vector size
MAX_ACTION_VECTOR_SIZE = 100      # Maximum reasonable action vector size

# =============================================================================
# PHASE 3 IMPLEMENTATION CONSTANTS (HARDCODED VALUE ELIMINATION)
# =============================================================================

# Model Loading Constants
# Seconds to wait between model loading retries
RL_MODEL_LOAD_RETRY_DELAY = 1.0
# Index for traffic light definition array access
TRAFFIC_LIGHT_DEFINITION_INDEX = 0

# Memory Management Constants
# Maximum inference time records to keep
RL_INFERENCE_TIME_MAX_HISTORY = 1000
# Number of recent records to keep after cleanup
RL_INFERENCE_TIME_KEEP_RECENT = 100

# Logging Interval Constants
# Steps between action distribution logging
RL_ACTION_DISTRIBUTION_LOG_INTERVAL = 10

# Training Utility Constants
# Default number of checkpoints to keep
CLEANUP_DEFAULT_KEEP_LATEST = 5
MODEL_SIZE_CONVERSION_FACTOR = 1024                # Bytes to KB/MB conversion
# Minimum models needed for progress analysis
TRAINING_PROGRESS_MIN_MODELS = 2
# Assumed efficiency for parallel training
PARALLEL_TRAINING_EFFICIENCY = 0.8
# Seconds to minutes conversion
TIME_CONVERSION_MINUTES = 60
# Seconds to hours conversion
TIME_CONVERSION_HOURS = 3600
# Default reference time per timestep (seconds)
DEFAULT_TIME_PER_TIMESTEP = 0.01

# Model Training Constants
# PPO policy type for traffic control
TRAINING_POLICY_TYPE = "MlpPolicy"
# Activation function for neural networks
TRAINING_ACTIVATION_FUNCTION = nn.ReLU
# Verbose level for training output
TRAINING_VERBOSE_LEVEL = 1
# Efficiency baseline for single environment
PARALLEL_ENV_EFFICIENCY_BASELINE = 1.0
# Power for variance to std deviation conversion
VARIANCE_CALCULATION_POWER = 0.5

# =============================================================================
# TREE METHOD INTEGRATION CONSTANTS
# =============================================================================

# Traffic Flow Theory (from Tree Method shared/config.py)
TREE_METHOD_MAX_DENSITY = 150                    # vehicles per km per lane
TREE_METHOD_MIN_VELOCITY = 3                     # km/h minimum speed
# speed-density relationship parameter
TREE_METHOD_M_PARAMETER = 0.8
# speed-density relationship parameter
TREE_METHOD_L_PARAMETER = 2.8
# Tree Method calculation interval
TREE_METHOD_ITERATION_TIME_MINUTES = 1.5

# State Space Enhancement Constants
# Original 4 + 6 Tree Method features
RL_ENHANCED_EDGE_FEATURES_COUNT = 10
# Original 2 + 4 Tree Method features
RL_ENHANCED_JUNCTION_FEATURES_COUNT = 6
RL_NETWORK_LEVEL_FEATURES_COUNT = 5               # Global network metrics

# Traffic Engineering Thresholds for Normalization
# Maximum expected time loss per edge
MAX_TIME_LOSS_MINUTES = 10.0
# Maximum expected cost per edge
MAX_COST_PER_EDGE = 1000.0
MAX_FLOW_PER_LANE_PER_HOUR = 2000.0              # Maximum realistic flow
# Maximum vehicles per edge for normalization
MAX_VEHICLE_COUNT_PER_EDGE = 50
# Seconds for speed trend calculation
MOVING_AVERAGE_WINDOW_SIZE = 30

# Updated State Vector Size Estimates
# Edge count is now calculated dynamically using formula: 2 × 2 × (2 × dimension × (dimension - 1))
# No hardcoded edge count - calculated per grid dimension in config.py
# RL_ENHANCED_STATE_VECTOR_SIZE_ESTIMATE = (
#     [REMOVED] - Edge count now calculated dynamically in config.py
# )  # State vector size calculated dynamically per grid dimension

# Individual Feature Toggle System
# =================================
# Edge Features (per edge)
# Original: current speed / max speed ratio
ENABLE_EDGE_SPEED_RATIO = True
# Original: vehicles per meter (unbounded, problematic)
ENABLE_EDGE_DENSITY_SIMPLE = False
# Original: vehicles per second (unbounded, problematic)
ENABLE_EDGE_FLOW_SIMPLE = False
# Original: binary waiting time > 30s
ENABLE_EDGE_CONGESTION_FLAG = True
# Tree Method: traffic flow theory density
ENABLE_EDGE_NORMALIZED_DENSITY = True
# Tree Method: normalized flow (redundant with density)
ENABLE_EDGE_NORMALIZED_FLOW = False
# Tree Method: speed < optimal speed flag
ENABLE_EDGE_IS_BOTTLENECK = True
# Tree Method: time loss vs optimal
ENABLE_EDGE_NORMALIZED_TIME_LOSS = True
# Tree Method: flow × time_loss (derived metric)
ENABLE_EDGE_NORMALIZED_COST = False
# Tree Method: speed change trend
ENABLE_EDGE_SPEED_TREND = True

# Junction Features (per junction)
# Original: current phase / total phases
ENABLE_JUNCTION_PHASE_NORMALIZED = True
# Original: remaining duration normalized
ENABLE_JUNCTION_DURATION_NORMALIZED = True
# Enhanced: placeholder (always 0.0)
ENABLE_JUNCTION_INCOMING_FLOW = False
# Enhanced: placeholder (always 0.0)
ENABLE_JUNCTION_OUTGOING_FLOW = False
# Enhanced: placeholder (always 0.0)
ENABLE_JUNCTION_UPSTREAM_BOTTLENECKS = False
# Enhanced: placeholder (always 0.0)
ENABLE_JUNCTION_DOWNSTREAM_BOTTLENECKS = False

# Network Features (global)
# Global: ratio of bottlenecked edges
ENABLE_NETWORK_BOTTLENECK_RATIO = True
ENABLE_NETWORK_COST_NORMALIZED = True            # Global: total network cost
ENABLE_NETWORK_VEHICLES_NORMALIZED = True        # Global: total vehicle count
ENABLE_NETWORK_AVG_SPEED_NORMALIZED = True       # Global: average network speed
ENABLE_NETWORK_CONGESTION_RATIO = True           # Global: congestion level

# Calculate dynamic feature counts based on enabled features


def _count_enabled_edge_features():
    return sum([
        ENABLE_EDGE_SPEED_RATIO,
        ENABLE_EDGE_DENSITY_SIMPLE,
        ENABLE_EDGE_FLOW_SIMPLE,
        ENABLE_EDGE_CONGESTION_FLAG,
        ENABLE_EDGE_NORMALIZED_DENSITY,
        ENABLE_EDGE_NORMALIZED_FLOW,
        ENABLE_EDGE_IS_BOTTLENECK,
        ENABLE_EDGE_NORMALIZED_TIME_LOSS,
        ENABLE_EDGE_NORMALIZED_COST,
        ENABLE_EDGE_SPEED_TREND
    ])


def _count_enabled_junction_features():
    return sum([
        ENABLE_JUNCTION_PHASE_NORMALIZED,
        ENABLE_JUNCTION_DURATION_NORMALIZED,
        ENABLE_JUNCTION_INCOMING_FLOW,
        ENABLE_JUNCTION_OUTGOING_FLOW,
        ENABLE_JUNCTION_UPSTREAM_BOTTLENECKS,
        ENABLE_JUNCTION_DOWNSTREAM_BOTTLENECKS
    ])


def _count_enabled_network_features():
    return sum([
        ENABLE_NETWORK_BOTTLENECK_RATIO,
        ENABLE_NETWORK_COST_NORMALIZED,
        ENABLE_NETWORK_VEHICLES_NORMALIZED,
        ENABLE_NETWORK_AVG_SPEED_NORMALIZED,
        ENABLE_NETWORK_CONGESTION_RATIO
    ])


# Dynamic feature counts (will be calculated at runtime)
# Currently: 6 features enabled
RL_DYNAMIC_EDGE_FEATURES_COUNT = _count_enabled_edge_features()
# Currently: 2 features enabled
RL_DYNAMIC_JUNCTION_FEATURES_COUNT = _count_enabled_junction_features()
# Currently: 5 features enabled
RL_DYNAMIC_NETWORK_FEATURES_COUNT = _count_enabled_network_features()

# Updated State Vector Size (Dynamic)
# RL_DYNAMIC_STATE_VECTOR_SIZE_ESTIMATE = (
#     [REMOVED] - Edge count now calculated dynamically in config.py
# )  # State vector size calculated dynamically per grid dimension

# Now set the training estimate to use dynamic size
# State vector size calculated dynamically per grid dimension in config.py
# RL_TRAINING_STATE_VECTOR_SIZE_ESTIMATE = [REMOVED] - calculated dynamically

# =============================================================================
# IMITATION LEARNING CONSTANTS
# =============================================================================

# Demonstration Collection
# Number of demo episodes to collect
DEMONSTRATION_COLLECTION_DEFAULT_SCENARIOS = 500
# Default variation config
DEMONSTRATION_DEFAULT_CONFIG_FILE = "configs/demo_variation_default.json"
# Match Tree Method's decision frequency
DEMONSTRATION_DECISION_INTERVAL_SECONDS = 10

# Behavioral Cloning Pre-training
# Supervised learning learning rate
PRETRAINING_LEARNING_RATE = 1e-3
# Batch size for behavioral cloning
PRETRAINING_BATCH_SIZE = 64
PRETRAINING_EPOCHS = 10                           # Number of training epochs
PRETRAINING_VALIDATION_SPLIT = 0.1                # Validation set proportion
PRETRAINING_VERBOSE = True                        # Print training progress

# Pre-trained Model Storage
# Directory for pre-trained models
PRETRAINED_MODEL_DIR = "models/pretrained"
# Default pre-trained model name
PRETRAINED_MODEL_NAME = "tree_method_pretrained.zip"
