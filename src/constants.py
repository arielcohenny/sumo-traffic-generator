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

# SUMO Routing Modes (from SUMO documentation)
SUMO_ROUTING_MODE_DEFAULT = 0  # Default routing
SUMO_ROUTING_MODE_AGGREGATED = 1  # GPS-like with smoothed travel times
SUMO_ROUTING_MODE_IGNORE_REROUTERS = 8  # Ignore rerouter changes
SUMO_ROUTING_MODE_AGGREGATED_IGNORE_REROUTERS = 9  # Combined mode

# TraCI Commands (for reference and validation)
TRACI_CMD_FIND_ROUTE = 0x86
TRACI_CMD_FIND_INTERMODAL_ROUTE = 0x87

# Rerouting intervals (seconds)
REALTIME_REROUTING_INTERVAL_SECONDS = 30
FASTEST_REROUTING_INTERVAL_SECONDS = 45
ATTRACTIVENESS_REROUTING_INTERVAL_SECONDS = 60

# Route improvement thresholds (percentages)
REALTIME_ROUTE_IMPROVEMENT_THRESHOLD_PCT = 10
FASTEST_ROUTE_IMPROVEMENT_THRESHOLD_PCT = 15
ATTRACTIVENESS_ROUTE_IMPROVEMENT_THRESHOLD_PCT = 12

# Edge metrics for route quality assessment (from SUMO documentation)
EDGE_CONGESTION_THRESHOLD_SPEED_MS = 5.0  # m/s, below this is congested
EDGE_HIGH_DENSITY_VEHICLE_COUNT = 20  # vehicles per edge indicating high density
EDGE_SIGNIFICANT_WAITING_TIME_SEC = 30  # seconds, significant queue delay

# Route quality and performance
MAX_ROUTE_CHANGES_PER_VEHICLE = 10
ROUTE_CACHE_TTL_SECONDS = 300
ENABLE_ROUTE_PERFORMANCE_TRACKING = True
ROUTE_PERFORMANCE_LOG_INTERVAL = 100

# SUMO-specific constraints (from documentation)
SUMO_ROUTE_CHANGE_INTERSECTION_RESTRICTION = "Routes can only be changed if vehicle is not within an intersection"
SUMO_ROUTE_FIRST_EDGE_REQUIREMENT = "First edge in new route must match vehicle's current location"

# Error handling - Program termination codes
ROUTING_ERROR_REALTIME_FAILED = "ROUTING_001"
ROUTING_ERROR_FASTEST_FAILED = "ROUTING_002"
ROUTING_ERROR_ATTRACTIVENESS_FAILED = "ROUTING_003"
ROUTING_ERROR_INVALID_ROUTE = "ROUTING_004"
ROUTING_ERROR_INTERSECTION_RESTRICTION = "ROUTING_005"  # SUMO-specific error
ROUTING_ERROR_STRATEGY_ASSIGNMENT = "ROUTING_006"
ROUTING_ERROR_XML_PARSING = "ROUTING_007"
ROUTING_ERROR_MISSING_DATA = "ROUTING_008"
ROUTING_ERROR_TRACI_FAILURE = "ROUTING_009"

# Error message templates for stderr output before sys.exit(1)
ROUTING_ERROR_MSG_TEMPLATE = "FATAL ERROR [{code}]: Routing strategy '{strategy}' failed for vehicle '{vehicle_id}': {reason}"
ROUTING_VALIDATION_ERROR_MSG = "FATAL ERROR [{code}]: Route validation failed for vehicle '{vehicle_id}': {details}"
SUMO_CONSTRAINT_ERROR_MSG = "FATAL ERROR [{code}]: SUMO routing constraint violated for vehicle '{vehicle_id}': {constraint}"
XML_PARSING_ERROR_MSG = "FATAL ERROR [{code}]: Vehicle strategy XML parsing failed: {details}"
MISSING_DATA_ERROR_MSG = "FATAL ERROR [{code}]: Required routing data missing: {details}"
TRACI_ERROR_MSG = "FATAL ERROR [{code}]: TraCI command failed for vehicle '{vehicle_id}': {command} - {reason}"

# Vehicle Types
DEFAULT_VEHICLE_TYPES = 'passenger 90 public 10'
DEFAULT_PASSENGER_VEHICLE_PCT = 90
DEFAULT_PUBLIC_VEHICLE_PCT = 10

# Route Patterns
DEFAULT_PASSENGER_ROUTES = 'in 30 out 30 inner 25 pass 15'
DEFAULT_PUBLIC_ROUTES = 'in 25 out 25 inner 35 pass 15'

# Public Transit Constants
SECONDS_IN_DAY = 86400
DEFAULT_VEHICLES_DAILY_PER_ROUTE = 124

# How much to amplify preferred patterns during rush hours
TEMPORAL_BIAS_STRENGTH = 1.5

# Percentage Constraints
MIN_PERCENTAGE = 0
MAX_PERCENTAGE = 100

# Departure Patterns
DEFAULT_DEPARTURE_PATTERN = 'uniform'
UNIFORM_DEPARTURE_PATTERN = 'uniform'  # For constraint checking

# Departure Pattern Time Constraints
FIXED_START_TIME_HOUR = 0.0    # Midnight start for non-uniform patterns

# Sentinel Values (to detect if user explicitly provided parameters)
SENTINEL_START_TIME_HOUR = -999.0  # Impossible start time to detect defaults
SENTINEL_END_TIME = -999           # Impossible end time to detect defaults

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
MAX_END_TIME = 86400  # 24 hours in seconds
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

# Traffic Light Strategy Constants
TL_STRATEGY_OPPOSITES = 'opposites'
TL_STRATEGY_INCOMING = 'incoming'
TL_STRATEGY_PARTIAL_OPPOSITES = 'partial_opposites'

# Minimum Lanes by Traffic Light Strategy
MIN_LANES_FOR_TL_STRATEGY = {
    TL_STRATEGY_OPPOSITES: 1,
    TL_STRATEGY_INCOMING: 1,
    TL_STRATEGY_PARTIAL_OPPOSITES: 2,  # Requires 2+ lanes
}

# Partial Opposites Phase Durations (DEPRECATED - kept for reference only)
# NOTE: Actual durations are calculated dynamically by convert_to_green_only_phases()
# All phases get equal duration: 90s / num_phases
# - Interior junctions (4 phases): 22.5s each
# - Corner junctions (2-3 phases): 45s or 30s each
# This ensures fair Fixed baseline where all movements get equal green time
PARTIAL_OPPOSITES_STRAIGHT_RIGHT_GREEN = 22.5  # seconds (DEPRECATED - not used)
PARTIAL_OPPOSITES_LEFT_UTURN_GREEN = 22.5      # seconds (DEPRECATED - not used)

# Actuated Traffic Light Parameters (matching original decentralized-traffic-bottlenecks repository)
# These parameters align with the original research implementation for comparative experiments
ACTUATED_MAX_GAP = 3.0              # seconds - maximum time gap between vehicles to extend phase
ACTUATED_DETECTOR_GAP = 1.0         # seconds - detector placement distance in seconds at max speed
ACTUATED_PASSING_TIME = 10.0        # seconds - vehicle headway estimate
ACTUATED_FREQ = 300                 # seconds - detector data aggregation frequency
ACTUATED_SHOW_DETECTORS = True      # boolean - visibility in GUI for debugging
ACTUATED_MIN_DUR = 10               # seconds - minimum phase duration
ACTUATED_MAX_DUR = 70               # seconds - maximum phase duration

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

DEFAULT_MORNING_PCT = 40
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
# PHASE TIME BOUNDARIES & ATTRACTIVENESS CONSTANTS
# =============================================================================

# RUSH HOUR TIME CONSOLIDATION (excluding SIX_PERIODS_):
RUSH_HOUR_MORNING_START = 6.0
RUSH_HOUR_MORNING_END = 10.0
RUSH_HOUR_EVENING_START = 16.0
RUSH_HOUR_EVENING_END = 20.0
NIGHT_LOW_START = 22

# Phase Multipliers for Attractiveness
PHASE_MORNING_PEAK_DEPART_MULTIPLIER = 1.4
PHASE_MORNING_PEAK_ARRIVE_MULTIPLIER = 0.7
PHASE_MIDDAY_OFFPEAK_DEPART_MULTIPLIER = 1.0
PHASE_MIDDAY_OFFPEAK_ARRIVE_MULTIPLIER = 1.0
PHASE_EVENING_PEAK_DEPART_MULTIPLIER = 0.7
PHASE_EVENING_PEAK_ARRIVE_MULTIPLIER = 1.5
PHASE_NIGHT_LOW_DEPART_MULTIPLIER = 0.4
PHASE_NIGHT_LOW_ARRIVE_MULTIPLIER = 0.4

# Attractiveness Value Range
MIN_ATTRACTIVENESS_VALUE = 1
MAX_ATTRACTIVENESS_VALUE = 20

# =============================================================================
# ROUTE GENERATION CONSTANTS
# =============================================================================

# Route Generation Limits
MAX_ROUTE_RETRIES = 20

# Simulation End Factors (fraction of total simulation time to use for departures)
SIMULATION_END_FACTOR = 0.9  # 90% for general patterns
SIMULATION_END_FACTOR_SIX_PERIODS = 0.95  # 95% for six periods pattern

# Night Period Distribution
# 25% in evening part (10pm-12am), 75% early morning
NIGHT_EVENING_RATIO = 0.25

# Six Periods Time Constants (in hours)
SIX_PERIODS_MORNING_START = 6.0
SIX_PERIODS_MORNING_END = 12.0
SIX_PERIODS_MORNING_RUSH_START = 7.5
SIX_PERIODS_MORNING_RUSH_END = 9.5
SIX_PERIODS_NOON_START = 12.0
SIX_PERIODS_NOON_END = 17.0
SIX_PERIODS_EVENING_RUSH_START = 17.0
SIX_PERIODS_EVENING_RUSH_END = 19.0
SIX_PERIODS_EVENING_START = 19.0
SIX_PERIODS_EVENING_END = 22.0
SIX_PERIODS_NIGHT_START = 22.0
SIX_PERIODS_NIGHT_END = 30.0  # Wraps to next day (6am)
SIX_PERIODS_EARLY_MORNING_END = 6.0

# Six Periods Weights (percentages)
SIX_PERIODS_MORNING_WEIGHT = 20
SIX_PERIODS_MORNING_RUSH_WEIGHT = 30
SIX_PERIODS_NOON_WEIGHT = 25
SIX_PERIODS_EVENING_RUSH_WEIGHT = 20
SIX_PERIODS_EVENING_WEIGHT = 4
SIX_PERIODS_NIGHT_WEIGHT = 1

# =============================================================================
# GEOMETRY & SPATIAL CONSTANTS
# =============================================================================

# Distance Thresholds
GEOMETRY_DISTANCE_THRESHOLD = 50  # Distance threshold for geometry calculations
ZONE_ADJACENCY_DISTANCE_THRESHOLD = 10  # Distance threshold for zone adjacency

# =============================================================================
# TRAFFIC GENERATION CONSTANTS (ADDITIONAL)
# =============================================================================

# Time Conversion Constants
SECONDS_TO_HOURS_DIVISOR = 3600  # Convert seconds to hours
HOURS_IN_DAY = 24  # Hours in a day (0-24 format)
SECONDS_IN_24_HOURS = 86400  # Seconds in 24 hours

# Route Pattern Constants
# Number of pairs in route pattern (4 patterns × 2 values each)
ROUTE_PATTERN_PAIRS_COUNT = 8
ROUTE_PATTERN_EXPECTED_PAIRS = 4  # Expected number of pattern pairs

# Vehicle Generation Constants
INITIAL_VEHICLE_ID_COUNTER = 0  # Starting counter for vehicle IDs
SINGLE_SAMPLE_COUNT = 1  # Number of samples to take when sampling single edges
EDGE_SAMPLE_SLICE_LIMIT = 5  # Limit for edge sample display
# PERCENTAGE_TO_DECIMAL_DIVISOR = 100.0  # Convert percentage to decimal

# Default Fallback Values
DEFAULT_DEPARTURE_TIME_FALLBACK = 0  # Fallback departure time when list is empty
DEFAULT_ROUTE_WEIGHT = 1.0  # Default weight for uniform route distributions
DEFAULT_REST_WEIGHT = 10  # Default rest weight for rush hours pattern
MINIMUM_ROUTE_COUNT = 1  # Minimum routes per pattern
MINIMUM_VEHICLES_PER_ROUTE = 1.0  # Minimum vehicles per route

# Mathematical Constants
TEMPORAL_BIAS_INVERSE_FACTOR = 2.0  # Factor for inverse temporal bias calculation

# Array/List Access Constants
ARRAY_FIRST_ELEMENT_INDEX = 0  # Index for accessing first element in arrays/lists
# Step increment for parsing pairs (pattern + percentage)
RANGE_STEP_INCREMENT = 2
SINGLE_INCREMENT = 1  # Single increment for counters

# Route Generation Numerical Constants
ROUTE_ID_INCREMENT = 1  # Increment for route ID counter
VEHICLE_INCREMENT = 1  # Increment for vehicle counter
VEHICLES_FOR_ROUTE_INCREMENT = 1  # Increment for vehicles per route counter

# Default Attribute Values
# Default value for edge attributes when missing
DEFAULT_EDGE_ATTRIBUTE_VALUE = 0.0
FALLBACK_ATTRIBUTE_VALUE = 0.0  # Fallback value when attribute is None

# Time Range Constants
TIME_RANGE_START = 0  # Start time for ranges (0 hours)
TIME_RANGE_END_24H = 24  # End time for 24-hour ranges
EXAMPLE_START_TIME_8AM = 8.0  # Example start time (8:00 AM)
# EXAMPLE_TIME_3600_SECONDS = 3600  # Example time (1 hour in seconds)
EXAMPLE_RESULT_28800 = 28800  # Example result (8:00 AM in seconds)
EXAMPLE_RESULT_32400 = 32400  # Example result (9:00 AM in seconds)

# Comment Reference Numbers (for documentation)
ROUTES_MD_LINE_18 = 18  # Reference to ROUTES2.md line 18
ROUTES_MD_LINE_99 = 99  # Reference to ROUTES2.md line 99
ATTRACTIVENESS_PHASES_COUNT = 4  # Number of attractiveness time phases

# =============================================================================
# STRING LITERALS AND IDENTIFIERS
# =============================================================================

# Phase Names
PHASE_MORNING_PEAK = "morning_peak"
PHASE_MIDDAY_OFFPEAK = "midday_offpeak"
PHASE_EVENING_PEAK = "evening_peak"
PHASE_NIGHT_LOW = "night_low"

# Route Pattern Names
PATTERN_IN = "in"
PATTERN_OUT = "out"
PATTERN_INNER = "inner"
PATTERN_PASS = "pass"

# Traffic Direction Names
DIRECTION_DEPART = "depart"
DIRECTION_ARRIVE = "arrive"

# Attribute Suffixes and Prefixes
SUFFIX_ATTRACTIVENESS = "_attractiveness"
ATTR_PREFIX_PHASE_DEPART = "_depart_attractiveness"
ATTR_PREFIX_PHASE_ARRIVE = "_arrive_attractiveness"

# Vehicle and Route Identifiers
VEHICLE_ID_PREFIX = "veh"
ROUTE_ID_PREFIX = "route"

# Vehicle Types
VEHICLE_TYPE_PASSENGER = "passenger"
VEHICLE_TYPE_PUBLIC = "public"

# Routing Strategy Names
ROUTING_SHORTEST = "shortest"
ROUTING_REALTIME = "realtime"
ROUTING_FASTEST = "fastest"
ROUTING_ATTRACTIVENESS = "attractiveness"

# Departure Pattern Names
DEPARTURE_PATTERN_UNIFORM = "uniform"
DEPARTURE_PATTERN_SIX_PERIODS = "six_periods"
DEPARTURE_PATTERN_RUSH_HOURS = "rush_hours"

# Six Periods Names
PERIOD_MORNING = "morning"
PERIOD_MORNING_RUSH = "morning_rush"
PERIOD_NOON = "noon"
PERIOD_EVENING_RUSH = "evening_rush"
PERIOD_EVENING = "evening"
PERIOD_NIGHT = "night"

# Rush Hours Pattern Components
RUSH_HOURS_PREFIX = "rush_hours:"
RUSH_HOURS_REST = "rest"
RUSH_HOURS_SEPARATOR = ":"

# XML and Data Attribute Names
ATTR_NAME = "name"
ATTR_WEIGHT = "weight"
ATTR_ID = "id"
ATTR_START = "start"
ATTR_END = "end"
ATTR_EDGE = "edge"
ATTR_DEPART = "depart"
ATTR_ATTRACTIVENESS = "attractiveness"
ATTR_CURRENT_PHASE = "current_phase"
ATTR_TYPE = "type"
ATTR_FROM_EDGE = "from_edge"
ATTR_TO_EDGE = "to_edge"
ATTR_ROUTE_EDGES = "route_edges"
ATTR_ROUTING_STRATEGY = "routing_strategy"

# Network Function Types
FUNCTION_INTERNAL = "internal"

# Function/Field Names
FIELD_PASSENGER_ROUTES = "passenger_routes"
FIELD_PUBLIC_ROUTES = "public_routes"

# Default Route Pattern Strings
DEFAULT_PASSENGER_ROUTE_PATTERN = "in 30 out 30 inner 25 pass 15"
DEFAULT_PUBLIC_ROUTE_PATTERN = "in 25 out 25 inner 35 pass 15"

# Edge Classification Names
EDGE_TYPE_BOUNDARY = "boundary"
EDGE_TYPE_INNER = "inner"

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Network Validation Thresholds
MIN_XML_FILE_SIZE = 100  # Minimum reasonable size for XML files (bytes)
MIN_PHASE_DURATION = 1  # Minimum traffic light phase duration (seconds)
MAX_PHASE_DURATION = 120  # Maximum traffic light phase duration (seconds)
MIN_CYCLE_TIME = 10  # Minimum traffic light cycle time (seconds)
MAX_CYCLE_TIME = 300  # Maximum traffic light cycle time (seconds)
MIN_GREEN_TIME_RATIO = 0.2  # Minimum green time as fraction of cycle time

# Argument Validation Limits
MAX_BLOCK_SIZE_VALIDATION = 1000  # Maximum block size for realism (meters)
MAX_NUM_VEHICLES_VALIDATION = 10000  # Maximum vehicles for performance

# Regular Expression Patterns
# Pattern for junction IDs (e.g., A1, B2, C10)
JUNCTION_ID_PATTERN = r"^[A-Z]+\d+$"
# Pattern for edge IDs (e.g., A1B1, B2C2)
EDGE_ID_PATTERN = r"^[A-Z]+\d+[A-Z]+\d+$"

# =============================================================================
# PROGRESS BAR VALUES
# =============================================================================

PROGRESS_START = 0
PROGRESS_PIPELINE_CREATED = 10
PROGRESS_EXECUTION_STARTED = 20
PROGRESS_EXECUTION_RUNNING = 50
PROGRESS_COMPLETED = 100
