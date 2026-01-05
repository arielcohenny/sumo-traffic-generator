# Initialization and Setup

## Data Directory Cleanup

- **Step**: Clean and prepare output directory
- **Function**: Directory cleanup via pipeline orchestration
- **Process**:
  - Remove existing `workspace/` directory if it exists (`shutil.rmtree(CONFIG.output_dir)`)
  - Create fresh `workspace/` directory (`CONFIG.output_dir.mkdir(exist_ok=True)`)
- **Purpose**: Ensures clean state for each simulation run

## Command-Line Argument Parsing

- **Step**: Parse and validate all CLI arguments
- **Function**: `src/args/parser.py` via CLI orchestration
- **Process**: Parse 20 available arguments with defaults and validation
- **Available Arguments**: See Available Arguments section below

## Seed Initialization

- **Step**: Initialize random seed for reproducible simulations
- **Function**: Seed generation via pipeline orchestration
- **Process**:
  - Use provided `--seed` value if specified
  - Generate random seed if not provided: `random.randint(0, 2**32 - 1)`
  - Print seed value for reproduction: `print(f"Using seed: {seed}")`

## Available Arguments

### `--grid_dimension` (float, default: 5)

Defines the grid's number of rows and columns for synthetic network generation.

### `--block_size_m` (int, default: 200)

Sets block size in meters for grid network generation.

### `--junctions_to_remove` (str, default: "0")

Number of junctions to remove or comma-separated list of specific junction IDs (e.g., "5" or "A0,B1,C2").

### `--lane_count` (str, default: "realistic")

Sets the lane count. 3 algorithms are available:

- `realistic`: Zone-based demand calculation
- `random`: Randomized within bounds (1-3 lanes)
- Integer value: Fixed count for all edges

### `--num_vehicles` (int, default: 300)

Total vehicles to generate.

### `--seed` (int, optional)

Controls randomization. If not provided, random seed is generated.

### `--step-length` (float, default: 1.0)

Simulation step length in seconds for TraCI control loop.

### `--end-time` (int, default: 7200)

Simulation duration in seconds.

### `--attractiveness` (str, default: "land_use")

Sets the departure and arrival attractiveness of each edge. Three methods available:

- `land_use`: Zone-based calculation (default)
- `poisson`: Random distribution
- `iac`: Intersection accessibility calculation

### `--start_time_hour` (float, default: 0.0)

Real-world hour when simulation starts (0-24) for temporal attractiveness. Used for 4-phase variation calculation.

### `--departure_pattern` (str, default: "uniform")

Vehicle departure timing. Three patterns available:

- `six_periods`: Research-based daily structure (requires 24h simulation starting at midnight)
- `uniform`: Even distribution (flexible start time and duration)
- `custom:HH:MM-HH:MM,percent;...`: Custom time windows with percentages (e.g., `custom:9:00-10:00,50;17:00-18:00,30`)

### `--routing_strategy` (str, default: "shortest 100")

Vehicle routing behavior. Four strategies with percentage mixing:

- `shortest`: Static shortest path
- `realtime`: 30-second dynamic rerouting
- `fastest`: 45-second fastest path rerouting
- `attractiveness`: Multi-criteria routing

### `--vehicle_types` (str, default: "passenger 90 public 10")

Vehicle type distribution. Two types with percentage assignment:

- `passenger`: Cars (5.0m length, 13.9 m/s max speed)
- `public`: Buses (10.0m length, 11.1 m/s max speed)

### `--traffic_light_strategy` (str, default: "partial_opposites")

Applied strategies for traffic lights. Three strategies available.

- `partial_opposites`: Straight+right and left+u-turn movements in separate phases (requires 2+ lanes)
- `opposites`: Opposing directions signal together
- `incoming`: Each edge gets separate phase

### `--traffic_control` (str, default: "tree_method")

Dynamic signal control. Five methods available:

- `tree_method`: Tree Method (Decentralized Bottleneck Prioritization Algorithm)
- `atlcs`: ATLCS (Adaptive Traffic Light Control System with enhanced bottleneck detection)
- `rl`: Reinforcement Learning (Deep Q-Network based adaptive control)
- `actuated`: SUMO gap-based control (industry-standard baseline)
- `fixed`: Static timing from configuration (simple baseline)

### `--tree-method-interval` (int, default: 90)

Tree Method calculation interval in seconds. Controls how often the Tree Method algorithm runs its optimization calculations.

**Performance Configuration:**

- **Lower values (30-60s)**: More responsive traffic control, higher CPU usage
- **Higher values (120-300s)**: More efficient computation, less responsive control
- **Default (90s)**: Balanced efficiency and responsiveness

**Valid Range:** 30-300 seconds

**Independence:** Tree Method timing is completely independent of traffic light cycle timing.

**Technical Implementation:** Overrides `TREE_METHOD_ITERATION_INTERVAL_SEC` constant in `src/config.py`.

**Examples:**

```bash
# Responsive control (every 60 seconds)
python -m src.cli --traffic_control tree_method --tree-method-interval 60

# Efficient control (every 2 minutes)
python -m src.cli --traffic_control tree_method --tree-method-interval 120

# High-performance scenarios (every 3 minutes)
python -m src.cli --traffic_control tree_method --tree-method-interval 180
```

### `--bottleneck-detection-interval` (int, default: 60)

Enhanced bottleneck detection interval in seconds for ATLCS. Controls how often the ATLCS enhanced bottleneck detector runs its analysis.

**Enhanced Bottleneck Detection Features:**

- **Advanced Metrics**: Uses density, speed, queue length, and waiting time (vs Tree Method's speed-only)
- **Predictive Analysis**: Identifies bottlenecks before they fully form
- **Multi-Criteria Assessment**: Combines multiple traffic indicators for robust detection
- **Real-Time Responsiveness**: More frequent updates than Tree Method's strategic intervals

**Performance Configuration:**

- **Lower values (30-45s)**: More responsive bottleneck detection, higher CPU usage
- **Higher values (90-120s)**: More efficient computation, less responsive detection
- **Default (60s)**: Balanced detection frequency and computational efficiency

**Valid Range:** 30-120 seconds

**Integration:** Works alongside Tree Method's 90-second strategic calculations to provide tactical bottleneck prevention.

### `--atlcs-interval` (int, default: 5)

ATLCS dynamic pricing update interval in seconds for ATLCS. Controls how often the ATLCS pricing engine calculates congestion-based pricing updates.

**ATLCS Dynamic Pricing Features:**

- **Congestion-Based Pricing**: Higher congestion severity receives higher priority scores
- **Signal Priority Calculation**: Converts pricing data to traffic light extension recommendations
- **Real-Time Adaptation**: Rapid response to changing traffic conditions
- **Bottleneck Prevention**: Extends green phases dynamically to prevent jam formation

**Performance Configuration:**

- **Lower values (1-3s)**: Maximum responsiveness, highest CPU usage
- **Higher values (10-15s)**: More efficient computation, reduced responsiveness
- **Default (5s)**: Optimal balance for real-time traffic light control

**Valid Range:** 1-15 seconds

**Technical Implementation:** Updates shared phase durations that Tree Method can access, enabling coordinated traffic control.

**Examples:**

```bash
# Responsive ATLCS configuration
python -m src.cli --traffic_control atlcs --bottleneck-detection-interval 45 --atlcs-interval 3

# Efficient ATLCS configuration
python -m src.cli --traffic_control atlcs --bottleneck-detection-interval 90 --atlcs-interval 10

# Balanced ATLCS with Tree Method coordination
python -m src.cli --traffic_control atlcs --tree-method-interval 90 --bottleneck-detection-interval 60 --atlcs-interval 5
```

### `--rl_model_path` (str, optional)

Path to trained reinforcement learning model for inference mode. When using `--traffic_control rl`, this parameter determines whether the RL controller runs in training or inference mode.

**Mode Selection:**

- **With `--rl_model_path`**: Inference mode using trained neural network for signal control decisions
- **Without `--rl_model_path`**: Training mode with random exploration actions (for model training)

**Model Format:** Stable-Baselines3 `.zip` checkpoint files containing DQN policy network

**Training Workflow:**

1. Train model using `scripts/train_rl_production.py`
2. Optional: Use imitation learning from Tree Method demonstrations
3. Load trained checkpoint for inference using `--rl_model_path`

**Examples:**

```bash
# Inference mode with trained model
python -m src.cli --traffic_control rl --rl_model_path models/checkpoint/rl_traffic_model_410000_steps.zip

# Training mode (random actions for exploration)
python -m src.cli --traffic_control rl --end-time 7200
```

**Documentation:** See `docs/RL_IMPLEMENTATION.md` and `docs/IMITATION_LEARNING_GUIDE.md` for complete training workflow.

### `--rl-cycle-lengths` (int list, default: [90])

List of traffic light cycle lengths (in seconds) that the RL agent can select from. Controls the action space for cycle length decisions.

**Configuration:**

- **Single cycle (default)**: `--rl-cycle-lengths 90` - Fixed 90-second cycles
- **Multiple cycles**: `--rl-cycle-lengths 60 90 120` - Agent can choose between 60s, 90s, or 120s
- **Extended range**: `--rl-cycle-lengths 60 75 90 105 120` - More granular control

**Selection Strategy:** Controlled by `--rl-cycle-strategy` parameter

**Action Space Impact:** More cycle options increase action space complexity, potentially requiring longer training

**Examples:**

```bash
# Fixed cycle length
python -m src.cli --traffic_control rl --rl-cycle-lengths 90

# Variable cycle lengths with random selection
python -m src.cli --traffic_control rl --rl-cycle-lengths 60 90 120 --rl-cycle-strategy random
```

### `--rl-cycle-strategy` (str, default: "fixed")

Strategy for selecting cycle length from the `--rl-cycle-lengths` options.

**Strategies:**

- **`fixed`** (default): Always use first cycle length from list (deterministic)
- **`random`**: Randomly select cycle length at each decision point (stochastic exploration)
- **`sequential`**: Cycle through lengths in order (deterministic pattern)

**Use Cases:**

- **Fixed**: Production inference with trained models (consistent behavior)
- **Random**: Training phase for exploration (discovering optimal cycle lengths)
- **Sequential**: Testing all cycle lengths systematically

**Note:** Only affects behavior when multiple cycle lengths are provided in `--rl-cycle-lengths`

**Examples:**

```bash
# Fixed strategy (use first cycle length only)
python -m src.cli --traffic_control rl --rl-cycle-lengths 90 --rl-cycle-strategy fixed

# Random strategy for training
python -m src.cli --traffic_control rl --rl-cycle-lengths 60 90 120 --rl-cycle-strategy random
```

### `--gui` (flag)

Launch SUMO GUI.

### `--land_use_block_size_m` (float, default: 25.0)

Zone cell size in meters for land use zone generation.

**Default**: 25.0m for both network types (following research paper methodology from "A Simulation Model for Intra-Urban Movements")

**Purpose**: Controls the resolution of land use zone generation. Creates fine-grained cells for detailed spatial analysis independent of street block size.

### `--tree_method_sample` (str, optional)

Path to folder containing pre-built Tree Method sample files for bypass mode.

**Purpose**: Enables testing and validation using original research networks without generating new networks.

**Behavior**:

- **Bypass Mode**: Skips Steps 1-8 entirely, goes directly to Step 9 (Dynamic Simulation)
- **File Requirements**: Folder must contain `network.net.xml`, `vehicles.trips.xml`, and `simulation.sumocfg.xml`
- **File Management**: Automatically copies and adapts sample files to our pipeline naming convention
- **Validation**: Tests our Tree Method implementation against established research benchmarks

**Incompatible Arguments**: Cannot be used with network generation arguments (`--grid_dimension`, `--block_size_m`, `--junctions_to_remove`, `--lane_count`)

**Usage Examples**:

```bash
# Basic Tree Method validation
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control tree_method --gui

# Compare traffic control methods on identical network
env PYTHONUNBUFFERED=1 python -m src.cli --tree_method_sample evaluation/datasets/networks/ --traffic_control actuated --gui
```

### `--custom_lanes` (str, optional)

Custom lane definitions for specific edges in synthetic grid networks.

**Format**: `"EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=..."`

**Supported Syntax**:

- **Full Specification**: `A1B1=tail:2,head:B1B0:1,B1C1:2` (custom tail + explicit head movements)
- **Tail-Only**: `A1B1=tail:2` (custom tail lanes, preserve existing movements)
- **Head-Only**: `A1B1=head:B1B0:1,B1C1:2` (automatic tail, custom movements)
- **Dead-End**: `A1B1=tail:2,head:` (create dead-end street)

**Multiple Edges**: Separate configurations with semicolons

**Constraints**:

- Synthetic networks only
- Edge IDs must match grid pattern (A1B1, B2C2, etc.)
- Lane counts must be 1-3
- Mutually exclusive with `--custom_lanes_file`

### `--custom_lanes_file` (str, optional)

File containing custom lane definitions for complex scenarios.

**Format**: Same syntax as `--custom_lanes`, one configuration per line
**Features**:

- Supports comments (lines starting with #)
- UTF-8 encoding required
- Line-specific error reporting
- Mutually exclusive with `--custom_lanes`

**Example File Content**:

```
# Custom lane configuration file
A1B1=tail:2,head:B1B0:1,B1C1:2
A2B2=tail:3,head:B2C2:2,B2B1:1
# Dead-end creation
D1E1=tail:2,head:
```

## Argument Validation

- **Step**: Validate all CLI arguments before processing
- **Function**: Argument validation (to be implemented)
- **Process**: Comprehensive validation of all input parameters
- **Validations Required**:

### Routing Strategy Validation

- **Current**: Implemented in `parse_routing_strategy()` in `src/traffic/routing.py`
- **Checks**:
  - Format validation: Must be pairs of strategy name + percentage
  - Valid strategies: {"shortest", "realtime", "fastest", "attractiveness"}
  - Percentage range: 0-100 for each strategy
  - Sum validation: Percentages must sum to exactly 100 (±0.01 tolerance)
  - Type validation: Percentage values must be valid floats

### Vehicle Types Validation

- **Current**: Implemented in `parse_vehicle_types()` in `src/traffic/vehicle_types.py`
- **Checks**:
  - Format validation: Must be pairs of vehicle type + percentage
  - Valid types: {"passenger", "public"}
  - Percentage range: 0-100 for each type
  - Sum validation: Percentages must sum to exactly 100 (±0.01 tolerance)
  - Type validation: Percentage values must be valid floats

### Departure Pattern Validation

- **Current**: Fully implemented
- **Implemented Checks**:
  - Valid pattern names: {"six_periods", "uniform", "custom:..."}
  - Format validation for custom patterns (HH:MM-HH:MM,percent;...)
  - Time window validation within simulation bounds (start_time to start_time + duration)
  - Percentage validation (total ≤ 100%)
  - Overlap detection between time windows
  - Time format validation (HH:MM 24-hour clock)

### Numeric Range Validations

- **Current**: Basic type checking only
- **Needed Checks**:
  - `--grid_dimension`: > 0, reasonable upper bound (e.g., ≤ 20)
  - `--block_size_m`: > 0, reasonable range (50-1000m)
  - `--num_vehicles`: > 0, reasonable upper bound
  - `--step-length`: > 0, reasonable range (0.1-10.0 seconds)
  - `--end-time`: > 0
  - `--start_time_hour`: 0-24 range
  - `--land_use_block_size_m`: > 0, reasonable range (10-100m). **Default**: 25.0m for both network types (research paper methodology)

### Junctions to Remove Validation

- **Current**: String input only
- **Needed Checks**:
  - Format validation for comma-separated junction IDs
  - Numeric validation when integer count provided
  - Range validation (can't exceed grid capacity)
  - Junction ID format validation for specific removal

### Lane Count Validation

- **Current**: String acceptance only
- **Needed Checks**:
  - Valid algorithm names: {"realistic", "random"}
  - Integer validation for fixed count mode
  - Range validation for fixed counts (1-5 lanes)

### Cross-Argument Validation

- **Current**: Not implemented
- **Needed Checks**:
  - Tree Method sample vs grid parameters (mutually exclusive usage)
  - Grid dimension vs junctions to remove capacity limits
  - Traffic light strategy compatibility with network type

### Choice Validations

- **Current**: Implemented via argparse choices
- **Existing Checks**:

  - `--attractiveness`: {"poisson", "land_use", "iac"}
  - `--traffic_light_strategy`: {"opposites", "incoming"}
  - `--traffic_control`: {"tree_method", "actuated", "fixed"}

- **Error Handling**: Use existing `ValidationError` class for consistent error reporting
