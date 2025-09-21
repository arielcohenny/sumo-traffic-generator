"""
RL-specific constants for traffic signal control.

Constants specific to reinforcement learning components including
training parameters, reward system configuration, and RL environment settings.
General simulation constants are in src/constants.py.
"""

# =============================================================================
# RL TRAINING CONSTANTS
# =============================================================================

# PPO Training Parameters
DEFAULT_LEARNING_RATE = 2e-4  # Conservative for expensive episodes
DEFAULT_CLIP_RANGE = 0.1      # Prevent large policy updates
DEFAULT_BATCH_SIZE = 1024     # Balance stability and memory
DEFAULT_N_STEPS = 2048        # Steps per update
DEFAULT_N_EPOCHS = 10         # Optimization epochs per update
DEFAULT_GAMMA = 0.99          # Discount factor
DEFAULT_GAE_LAMBDA = 0.95     # GAE parameter

# Training Duration
DEFAULT_TOTAL_TIMESTEPS = 100000
DEFAULT_CHECKPOINT_FREQ = 10000

# Parallel Execution
DEFAULT_N_PARALLEL_ENVS = 4        # Number of parallel environments for training
MIN_PARALLEL_ENVS = 1              # Minimum parallel environments
MAX_PARALLEL_ENVS = 16             # Maximum parallel environments (memory constraint)
PARALLEL_WORKSPACE_PREFIX = "rl_training"  # Base name for parallel workspace directories
SINGLE_ENV_THRESHOLD = 1           # Threshold for using DummyVecEnv vs SubprocVecEnv

# Model Paths
DEFAULT_MODEL_SAVE_PATH = "models/rl_traffic_policy"
DEFAULT_MODELS_DIRECTORY = "models"

# Phase 4: Training Pipeline Constants
TRAINING_NETWORK_ARCHITECTURE = [256, 256]        # Hidden layer dimensions for MLP
TRAINING_DEVICE_AUTO = "auto"                      # Device selection (auto, cpu, cuda)
TRAINING_TENSORBOARD_LOG_DIR = "tensorboard_logs" # TensorBoard logging directory
TRAINING_BEST_MODEL_PREFIX = "best_model"         # Prefix for best model saves
TRAINING_CHECKPOINT_PREFIX = "checkpoint"         # Prefix for checkpoint saves
TRAINING_MODEL_METADATA_EXTENSION = ".json"       # Extension for model metadata files
TRAINING_EVAL_EPISODES_PER_CHECKPOINT = 5         # Episodes to evaluate at each checkpoint
TRAINING_PATIENCE_EPISODES = 20                   # Episodes to wait before early stopping
TRAINING_MIN_IMPROVEMENT_THRESHOLD = 0.01         # Minimum improvement to reset patience

# =============================================================================
# RL ENVIRONMENT CONSTANTS
# =============================================================================

# State Space Configuration
STATE_NORMALIZATION_MIN = 0.0
STATE_NORMALIZATION_MAX = 1.0

# State Feature Normalization Thresholds
MAX_DENSITY_VEHICLES_PER_METER = 0.2    # Maximum expected density for normalization
MAX_FLOW_VEHICLES_PER_SECOND = 1.0      # Maximum expected flow for normalization
CONGESTION_WAITING_TIME_THRESHOLD = 30.0  # Seconds - waiting time threshold for congestion flag

# Action Space Configuration - Traffic Signal Phases
NUM_TRAFFIC_LIGHT_PHASES = 4            # Number of phases per intersection (0-3)
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
ACTIONS_PER_INTERSECTION = 2  # phase + duration
PHASE_ACTION_INDEX = 0
DURATION_ACTION_INDEX = 1

# Phase Duration Options (seconds)
PHASE_DURATION_OPTIONS = [10, 15, 20, 30, 45, 60, 90, 120]
MIN_PHASE_DURATION = 10
MAX_PHASE_DURATION = 120

# Traffic Signal Constraints (seconds)
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 120
YELLOW_CLEARANCE_TIME = 3

# Decision Making Frequency
DECISION_INTERVAL_SECONDS = 10  # Fixed interval decision making
MIN_DECISION_INTERVAL = 5
MAX_DECISION_INTERVAL = 30

# =============================================================================
# REWARD SYSTEM CONSTANTS
# =============================================================================

# Reward Components
VEHICLE_PENALTY_WEIGHT = 1.0      # Weight for individual vehicle penalties
THROUGHPUT_BONUS_WEIGHT = 10.0    # Weight for episode completion bonuses (α parameter)

# Vehicle Tracking
MEASUREMENT_INTERVAL_STEPS = 10   # Steps between reward measurements
CREDIT_ASSIGNMENT_WINDOW_STEPS = 50  # Time window for crediting decisions

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
DEFAULT_RL_MODEL_PATH = None                 # No model path by default (training mode)
RL_MODEL_EXTENSION = ".zip"                  # Stable-baselines3 model file extension
RL_MODEL_VALIDATION_TIMEOUT = 10.0          # Seconds to wait for model validation
RL_MODEL_LOAD_RETRIES = 3                    # Number of model loading retry attempts

# Action Execution
RL_ACTION_EXECUTION_TIMEOUT = 2.0            # Seconds for action execution
RL_SAFETY_CHECK_ENABLED = True              # Enable safety constraint validation
RL_MIN_GREEN_TIME_ENFORCEMENT = True        # Enforce minimum green time constraints
RL_MAX_DURATION_ENFORCEMENT = True          # Enforce maximum duration constraints

# Performance Monitoring
RL_INFERENCE_TIME_TRACKING = True           # Track model inference timing
RL_ACTION_DISTRIBUTION_TRACKING = True      # Track action selection patterns
RL_STATISTICS_COLLECTION_INTERVAL = 100     # Steps between statistics collection
RL_PERFORMANCE_LOGGING_ENABLED = True       # Enable performance logging

# Training vs Inference Mode
RL_TRAINING_MODE = "training"                # Training mode identifier
RL_INFERENCE_MODE = "inference"              # Inference mode identifier
RL_DEFAULT_MODE = RL_TRAINING_MODE           # Default mode when no model provided

# Error Handling
RL_CONTROLLER_ERROR_PREFIX = "RL_CONTROLLER" # Error message prefix
RL_MODEL_COMPATIBILITY_CHECK = True          # Validate model compatibility
RL_GRACEFUL_DEGRADATION = False              # Whether to fall back to fixed timing on errors

# =============================================================================
# RL TRAINING CONFIGURATION (FIXED VALUES FOR NETWORK-SPECIFIC TRAINING)
# =============================================================================

# Network Configuration (FIXED - cannot change after Phase 2 implementation begins)
RL_TRAINING_GRID_DIMENSION = 3          # 3×3 grid (9 intersections)
RL_TRAINING_BLOCK_SIZE_M = 150           # 150 meters (realistic urban blocks)
RL_TRAINING_JUNCTIONS_TO_REMOVE = 0     # Keep full topology for coordination learning

# Traffic Configuration (FIXED)
RL_TRAINING_NUM_VEHICLES = 150          # 150 vehicles (~16-17 per intersection)
RL_TRAINING_VEHICLE_TYPES = "passenger 90 public 10"  # Realistic mix
RL_TRAINING_END_TIME = 3600             # 3600 seconds (1 hour episodes)

# RL-Specific Configuration (FIXED)
RL_TRAINING_DECISION_INTERVAL = 10      # 10 seconds between RL decisions
RL_TRAINING_MEASUREMENT_INTERVAL = 10   # 10 simulation steps between reward measurements

# Derived Constants (calculated from fixed values above)
RL_TRAINING_NUM_INTERSECTIONS = RL_TRAINING_GRID_DIMENSION * RL_TRAINING_GRID_DIMENSION
RL_TRAINING_ESTIMATED_EDGES = 12        # Estimated for 3×3 grid (exact value determined at runtime)
RL_TRAINING_STATE_VECTOR_SIZE_ESTIMATE = RL_TRAINING_ESTIMATED_EDGES * 4 + RL_TRAINING_NUM_INTERSECTIONS * 2  # ~66 dimensions
RL_TRAINING_ACTION_VECTOR_SIZE = RL_TRAINING_NUM_INTERSECTIONS * 2  # 18 actions (9 intersections × 2 decisions)

# =============================================================================
# DEVELOPMENT AND DEBUGGING CONSTANTS
# =============================================================================

# Default Values for Initialization
DEFAULT_INITIAL_STEP = 0              # Initial simulation step
DEFAULT_INITIAL_TIME = 0              # Initial simulation time
DEFAULT_INITIAL_PENALTY = 0.0         # Initial penalty value
DEFAULT_FALLBACK_VALUE = 0.0          # Default fallback for missing data
DEFAULT_OBSERVATION_PADDING = 0.0     # Value for padding observation vectors
DEFAULT_STEP_LENGTH = 1.0             # Standard simulation step length (seconds)
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
RL_MODEL_LOAD_RETRY_DELAY = 1.0                    # Seconds to wait between model loading retries
TRAFFIC_LIGHT_DEFINITION_INDEX = 0                  # Index for traffic light definition array access

# Memory Management Constants
RL_INFERENCE_TIME_MAX_HISTORY = 1000               # Maximum inference time records to keep
RL_INFERENCE_TIME_KEEP_RECENT = 100                # Number of recent records to keep after cleanup

# Logging Interval Constants
RL_ACTION_DISTRIBUTION_LOG_INTERVAL = 10           # Steps between action distribution logging

# Training Utility Constants
CLEANUP_DEFAULT_KEEP_LATEST = 5                    # Default number of checkpoints to keep
MODEL_SIZE_CONVERSION_FACTOR = 1024                # Bytes to KB/MB conversion
TRAINING_PROGRESS_MIN_MODELS = 2                   # Minimum models needed for progress analysis
PARALLEL_TRAINING_EFFICIENCY = 0.8                 # Assumed efficiency for parallel training
TIME_CONVERSION_MINUTES = 60                       # Seconds to minutes conversion
TIME_CONVERSION_HOURS = 3600                       # Seconds to hours conversion
DEFAULT_TIME_PER_TIMESTEP = 0.01                   # Default reference time per timestep (seconds)

# Model Training Constants
import torch.nn as nn
TRAINING_POLICY_TYPE = "MlpPolicy"                 # PPO policy type for traffic control
TRAINING_ACTIVATION_FUNCTION = nn.ReLU             # Activation function for neural networks
TRAINING_VERBOSE_LEVEL = 1                         # Verbose level for training output
PARALLEL_ENV_EFFICIENCY_BASELINE = 1.0             # Efficiency baseline for single environment
VARIANCE_CALCULATION_POWER = 0.5                   # Power for variance to std deviation conversion