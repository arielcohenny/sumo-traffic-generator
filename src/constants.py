"""
Centralized application constants.

Single source of truth for defaults, constraints, and configuration values
used by both CLI and GUI systems. This eliminates duplication across:
- src/args/parser.py (CLI defaults)
- src/ui/parameter_widgets.py (GUI widget defaults and constraints)
- src/ui/streamlit_app.py (GUI app defaults)
- src/config.py (config defaults)
"""

# =============================================================================
# NETWORK GENERATION CONSTANTS
# =============================================================================

# Grid Dimension
DEFAULT_GRID_DIMENSION = 5
MIN_GRID_DIMENSION = 1
MAX_GRID_DIMENSION = 20

# Block Size
DEFAULT_BLOCK_SIZE_M = 200
MIN_BLOCK_SIZE_M = 50
MAX_BLOCK_SIZE_M = 500
STEP_BLOCK_SIZE_M = 25

# Junctions
DEFAULT_JUNCTIONS_TO_REMOVE = '0'

# Lane Configuration
DEFAULT_LANE_COUNT = 'realistic'
DEFAULT_FIXED_LANE_COUNT = 2
MIN_LANE_COUNT = 1
MAX_LANE_COUNT = 5

# =============================================================================
# TRAFFIC GENERATION CONSTANTS
# =============================================================================

# Vehicle Numbers
DEFAULT_NUM_VEHICLES = 300
MIN_NUM_VEHICLES = 1
MAX_NUM_VEHICLES = 10000
STEP_NUM_VEHICLES = 50

# Routing Strategy
DEFAULT_ROUTING_STRATEGY = 'shortest 100'
DEFAULT_SHORTEST_ROUTING_PCT = 100
DEFAULT_REALTIME_ROUTING_PCT = 0
DEFAULT_FASTEST_ROUTING_PCT = 0
DEFAULT_ATTRACTIVENESS_ROUTING_PCT = 0

# Vehicle Types
DEFAULT_VEHICLE_TYPES = 'passenger 60 commercial 30 public 10'
DEFAULT_PASSENGER_VEHICLE_PCT = 60
DEFAULT_COMMERCIAL_VEHICLE_PCT = 30
DEFAULT_PUBLIC_VEHICLE_PCT = 10

# Percentage Constraints
MIN_PERCENTAGE = 0
MAX_PERCENTAGE = 100

# Departure Patterns
DEFAULT_DEPARTURE_PATTERN = 'uniform'

# =============================================================================
# SIMULATION CONSTANTS
# =============================================================================

# Seed
DEFAULT_SEED = 42
MIN_SEED = 1
MAX_SEED = 999999

# Step Length
DEFAULT_STEP_LENGTH = 1.0
MIN_STEP_LENGTH = 1.0
MAX_STEP_LENGTH = 10.0
STEP_LENGTH_STEP = 1.0

# End Time (Duration)
DEFAULT_END_TIME = 7200  # 2 hours in seconds
MIN_END_TIME = 1
MAX_END_TIME = 172800  # 48 hours in seconds
STEP_END_TIME = 3600  # 1 hour increments

# =============================================================================
# ZONE & ATTRACTIVENESS CONSTANTS
# =============================================================================

# Land Use Block Size
DEFAULT_LAND_USE_BLOCK_SIZE_M = 25.0
MIN_LAND_USE_BLOCK_SIZE_M = 10.0
MAX_LAND_USE_BLOCK_SIZE_M = 100.0
STEP_LAND_USE_BLOCK_SIZE_M = 5.0

# Attractiveness
DEFAULT_ATTRACTIVENESS = 'land_use'

# Time Dependency
DEFAULT_START_TIME_HOUR = 0.0
MIN_START_TIME_HOUR = 0.0
MAX_START_TIME_HOUR = 24.0
STEP_START_TIME_HOUR = 0.5

# =============================================================================
# TRAFFIC CONTROL CONSTANTS
# =============================================================================

# Traffic Light Strategy
DEFAULT_TRAFFIC_LIGHT_STRATEGY = 'opposites'

# Traffic Control Method
DEFAULT_TRAFFIC_CONTROL = 'tree_method'

# Algorithm Intervals
DEFAULT_BOTTLENECK_DETECTION_INTERVAL = 10
MIN_BOTTLENECK_DETECTION_INTERVAL = 1
MAX_BOTTLENECK_DETECTION_INTERVAL = 60

DEFAULT_ATLCS_INTERVAL = 5
MIN_ATLCS_INTERVAL = 1
MAX_ATLCS_INTERVAL = 30

DEFAULT_TREE_METHOD_INTERVAL = 90
MIN_TREE_METHOD_INTERVAL = 30
MAX_TREE_METHOD_INTERVAL = 300
STEP_TREE_METHOD_INTERVAL = 10

# =============================================================================
# RUSH HOURS PATTERN CONSTANTS
# =============================================================================

DEFAULT_MORNING_START = 7
DEFAULT_MORNING_END = 9
DEFAULT_MORNING_PCT = 40
DEFAULT_EVENING_START = 17
DEFAULT_EVENING_END = 19
DEFAULT_EVENING_PCT = 30
DEFAULT_REST_PCT = 30

# =============================================================================
# UI CONSTANTS
# =============================================================================

# Chart Display
DEFAULT_CHART_HEIGHT = 400

# Error Log Display  
ERROR_LOG_TRUNCATE_LIMIT = 2000

# File Handling
TEMP_FILE_PREFIX = "/tmp/custom_lanes_"

# =============================================================================
# SUMO INTEGRATION CONSTANTS
# =============================================================================

# Output Directory (parent directory where 'workspace' folder will be created)
DEFAULT_WORKSPACE_DIR = "."

# File Names
STATISTICS_FILE = "sumo_statistics.xml"
ERROR_LOG_FILE = "sumo_errors.log"
TRIPINFO_FILE = "sumo_tripinfo.xml"

# =============================================================================
# PROGRESS BAR VALUES
# =============================================================================

PROGRESS_START = 0
PROGRESS_PIPELINE_CREATED = 10
PROGRESS_EXECUTION_STARTED = 20
PROGRESS_EXECUTION_RUNNING = 50
PROGRESS_COMPLETED = 100