# Dynamic Simulation with Traffic Control

## Purpose and Process Overview

**Purpose of Dynamic Simulation:**

- **Traffic Flow Execution**: Runs the actual vehicle simulation using all generated network and traffic components
- **Real-Time Control**: Applies dynamic traffic control algorithms to optimize signal timing during simulation
- **Performance Measurement**: Collects traffic metrics (travel times, completion rates, throughput) for analysis
- **Research Platform**: Enables controlled comparison of different traffic control methods under identical conditions

## Dynamic Simulation Process

- **Step**: Execute SUMO simulation with real-time traffic control integration
- **Function**: `SumoController.run()` in `src/sumo_integration/sumo_controller.py`
- **Arguments Used**: `--traffic_control`, `--gui`, `--step_length`, `--end_time`, `--time_dependent`, `--start_time_hour`, `--routing_strategy`
- **Input Files**:
  - `workspace/grid.sumocfg` (SUMO configuration file)
  - `workspace/grid.net.xml` (complete network with attractiveness)
  - `workspace/vehicles.rou.xml` (vehicle routes and types)
  - `workspace/zones.poly.xml` (zones for visualization)

### TraCI Controller Initialization

**Controller Setup:**

- **Purpose**: Establishes Python-SUMO communication bridge for real-time control
- **Implementation**: `SumoController` class with per-step callback system
- **Features**:
  - Step-by-step simulation control
  - Real-time traffic light manipulation
  - Dynamic vehicle rerouting for realtime/fastest strategies
  - Traffic metrics collection throughout simulation

### Traffic Control Method Integration

**Conditional Object Initialization:**

- **Performance Optimization**: Only loads traffic control objects needed for selected method
- **Method-Specific Setup**: Different initialization paths based on `--traffic_control` argument

## Traffic Control Methods

### Tree Method (Decentralized Bottleneck Prioritization Algorithm)

**Algorithm Overview:**

- **Research Foundation**: Decentralized traffic control strategy based on congestion tree identification and cost calculation
- **Core Principle**: Addresses conflicting traffic flows that compete on opposing cycle times during specific phases at traffic intersections
- **Methodology**: Identifies and prioritizes congestion bottlenecks based on their global network influence rather than local impact
- **Decision Making**: Each intersection makes decentralized decisions using tree-shaped congestion analysis
- **Optimization Goal**: Minimize overall network congestion by prioritizing bottlenecks with highest global cost

**Theoretical Foundation:**

**Multi-Stage Tree Method Process:**

**Stage I - Network Representation and Pre-calculations:**

- **Network Transformation**: Divides street segments into body links (main road segments between intersections) and head links (final approach lanes leading into intersections)
- **Body Links**: Represent continuous traffic flow approaching junctions, capturing traffic buildup over time and critical for identifying congestion patterns
- **Head Links**: Shorter segments where vehicles queue before entering intersections, crucial for determining green time allocation
- **Network Graph**: Transforms urban area into network where nodes represent intersections/lane mergers and links represent street segments

**Stage II - Link Properties Pre-calculation:**

- **Traffic Parameters**: Establishes number of lanes and maximum travel speed (vf) for each body link
- **Flow Capacity**: Calculates maximum flow Qmax using fundamental traffic law (q(t) = v(t) \* k(t)) and May's formula
- **May's Equation**: v(t)/vf^(1-m) = 1 - (k(t)/kj)^(l-1) where kj = 150 veh/km, m = 0.8, l = 2.8
- **Maximum Flow Speed**: Derives Vqmax representing speed at which flow is maximized for each link

**Stage III - Real-Time Congestion Assessment:**

- **Cycle-Based Analysis**: Monitors speed on each body link during every traffic light cycle
- **Congestion Criteria**: Link classified as congested if observed speed < Vqmax, flowing if speed ≥ Vqmax
- **Dynamic Evaluation**: Real-time analysis conducted continuously throughout simulation

**Stage IV - Congestion Tree Formation:**

- **Trunk Identification**: Body links with head links leading to intersection identified as tree trunks (roots)
- **Recursive Construction**: Congestion trees built by adding adjacent congested body links feeding into trunk, continuing until all feeding links are non-congested
- **Tree Membership**: Body links may belong to multiple trees but serve as trunk for only one tree
- **Branch Classification**: All body links in tree classified as branches regardless of role

**Stage V - Tree Cost Calculation:**

- **Delay-Based Costing**: Cost represents additional time required to cross road compared to maximum flow conditions
- **Cost Formula**: C(t) = dij _ (1/v(t) - 1/vqmax) _ (q(t)*N*T)/60
  - dij: link length (km)
  - q(t): current flow on link
  - v(t): current speed on link
  - vqmax: speed under maximum flow conditions
  - N: number of lanes
  - T: traffic light cycle time (minutes)
- **Branch Cost Distribution**: Link cost divided by number of trees containing that link
- **Tree Total Cost**: Sum of all constituent branch costs, measured in vehicle hours (VH)

**Stage VI - Phase Cost Assignment:**

- **Phase-Link Mapping**: Each phase facilitates movement along specific body and head links
- **Cost Distribution**: Congested body link costs evenly distributed among all trees incorporating the link
- **Trunk Assignment**: Tree costs assigned to their respective trunks
- **Weight Calculation**: Head links receive weights (0-1) based on traffic volume handled during last cycle, normalized against other head links on same body link
- **Phase Cost**: Sum of corresponding body link costs multiplied by head link weights

**Stage VII - Dynamic Phase Duration Calculation:**

- **Competitive Allocation**: Fixed cycle length and phase order with dynamic duration distribution
- **Cost-Based Adjustment**: Phase durations for next cycle determined by current cycle performance
- **Proportional Distribution**: Available duration divided among phases according to their respective costs
- **Balancing Mechanism**: Higher cost phases receive proportionally longer durations to address congestion

**Implementation Process:**

- **Network JSON Generation**: Converts SUMO network to JSON format for algorithm processing
- **Tree Structure Loading**: Builds network tree and run configuration using `load_tree()`
- **Graph Construction**: Creates algorithm Graph object with network topology
- **Cycle Time Calculation**: Determines optimal signal cycle timing based on network characteristics

**Runtime Behavior:**

- **Step Frequency**: Updates traffic light states every simulation step
- **Algorithm Execution**: Calls `graph.update_traffic_lights()` with current time and cycle parameters
- **Phase Translation**: Converts algorithm decisions to SUMO traffic light color strings based on cost calculations
- **Signal Application**: Pushes new traffic light states to SUMO via TraCI

**Key Advantages:**

- **Global Optimization**: Prioritizes traffic flows based on global network cost rather than local impact
- **Real-Time Adaptability**: Analytical simplicity enables swift real-time adjustments in each cycle
- **Decentralized Operation**: Each intersection operates independently while considering network-wide effects
- **Bottleneck Focus**: Accurately identifies root causes of traffic congestion and upstream impacts
- **Fixed Cycle Benefits**: Maintains predictable phase ordering to avoid driver confusion while optimizing durations

**Setup Requirements:**

- **Files**: Network JSON, tree configuration, graph structure
- **Objects**: `Network`, `Graph`, cycle time calculation
- **Validation**: Runtime verification of algorithm behavior at configured frequency

### Actuated Control (SUMO Built-in)

**Algorithm Overview:**

- **SUMO Native**: Uses SUMO's built-in actuated traffic control system
- **Gap-Based Logic**: Extends green phases when vehicles are detected, switches when gaps occur
- **Sensor Simulation**: Simulates inductive loop detectors at intersection approaches
- **Adaptive Timing**: Adjusts signal timing based on real-time traffic detection

**Implementation Process:**

- **Minimal Setup**: No additional algorithm objects required
- **Automatic Operation**: SUMO handles all signal logic internally
- **Configuration**: Uses signal timings and detector positions from network generation

**Runtime Behavior:**

- **Autonomous Control**: No per-step intervention required from TraCI controller
- **Gap Detection**: Automatically detects vehicle presence and absence
- **Phase Extension**: Extends green phases when vehicles present, minimum/maximum timing constraints apply

### Fixed Timing Control (Static)

**Algorithm Overview:**

- **Static Timing**: Uses predefined, fixed-duration signal phases throughout simulation
- **Predictable Operation**: No adaptation to traffic conditions, consistent timing patterns
- **Baseline Comparison**: Serves as control group for comparing adaptive methods

**Implementation Process:**

- **No Setup**: Uses static timing from traffic light file generation
- **Grid Configuration**: Consistent timing across all intersections for fair comparison

**Runtime Behavior:**

- **No Intervention**: No per-step control required
- **Fixed Cycles**: Signal phases follow predetermined durations regardless of traffic
- **Predictable Patterns**: Enables controlled experimental conditions

## Dynamic Routing Integration

**Routing Strategy Execution:**

- **Static Strategies**: Shortest and attractiveness routes remain unchanged during simulation
- **Dynamic Strategies**: Realtime (30s) and fastest (45s) strategies trigger route updates via TraCI
- **Update Mechanism**: Controller tracks vehicle strategies and applies rerouting at specified intervals
- **Route Optimization**: Uses current edge travel times for dynamic route calculation

## Traffic Metrics Collection

**Performance Measurement:**

- **Real-Time Tracking**: Collects traffic statistics throughout simulation
- **Key Metrics**:
  - **Vehicle Arrivals**: Count of vehicles reaching destinations
  - **Vehicle Departures**: Count of vehicles entering simulation
  - **Completion Rate**: Percentage of vehicles successfully completing trips
  - **Average Travel Time**: Mean time from departure to arrival
  - **Throughput**: Vehicle flow rates through network

**Experimental Output:**

- **Metrics Display**: Prints experiment metrics at simulation completion
- **Research Format**: Structured output suitable for comparative analysis
- **Statistical Analysis**: Enables comparison between traffic control methods

## Dynamic Simulation Validation

**Runtime Verification:**

- **Algorithm Behavior**: Validates traffic control decisions during simulation
- **Performance Monitoring**: Tracks simulation health and progress
- **Error Handling**: Graceful handling of simulation errors and edge cases

## Dynamic Simulation Completion

- **Step**: Confirm successful simulation execution
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Simulation completed successfully."
- **Output**: Complete traffic simulation with performance metrics
- **Research Results**: Traffic control method performance data ready for analysis