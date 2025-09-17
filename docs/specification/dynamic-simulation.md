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
- **Arguments Used**: `--traffic_control`, `--gui`, `--step_length`, `--end_time`, `--start_time_hour`, `--routing_strategy`
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

- **Traffic Light Updates**: Updates traffic light states every simulation step (default: 1 second)
- **Algorithm Execution Timing**: Tree Method calculations occur every `tree_method_interval` seconds (default: 90 seconds, configurable via `--tree-method-interval`)
- **Calculation Independence**: Tree Method timing is completely independent of traffic light cycle timing for optimal performance
- **Performance Configuration**: 
  - Default 90 seconds balances efficiency with responsiveness
  - Lower intervals (30-60s) provide more responsive control with higher CPU usage
  - Higher intervals (120-300s) improve efficiency but reduce responsiveness
- **Algorithm Execution**: Calls `graph.calculate_iteration()` and `graph.update_traffic_lights()` with current time and interval parameters
- **Phase Translation**: Converts algorithm decisions to SUMO traffic light color strings based on cost calculations
- **Signal Application**: Pushes new traffic light states to SUMO via TraCI
- **Technical Implementation**: Uses `TREE_METHOD_ITERATION_INTERVAL_SEC` constant in `src/config.py` (overridable via CLI)

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

### ATLCS (Adaptive Traffic Light Control System)

**Algorithm Overview:**

- **Research Foundation**: Adaptive Traffic Light Control System implementing enhanced bottleneck detection and ATLCS through tactical bottleneck prevention
- **Core Principle**: Extends Tree Method's strategic network optimization with real-time responsive traffic light control
- **Coordination Architecture**: Junction-level handoff mechanism based on Tree Method's multi-edge tree detection for conflict-free operation
- **Tactical Focus**: Dynamic pricing theory applied to traffic control - higher congestion severity receives priority treatment
- **Real-Time Intervention**: Rapid response system preventing bottleneck formation through intelligent green phase extensions

**Theoretical Foundation:**

**ATLCS-Tree Method Coordination:**

- **Tree Method Role**: Strategic network optimization with 90-second calculation intervals for network-wide phase duration optimization
- **ATLCS Role**: Tactical bottleneck prevention with 5-60 second intervals for real-time green phase extensions
- **Coordination Logic**: Junction-level handoff based on multi-edge tree detection prevents method conflicts
- **Control Handoff**: Tree Method claims exclusive control when detecting multi-edge trees (complex bottlenecks), releases control when only single-edge trees remain
- **Complementary Operation**: Methods work as collaborative specialists rather than competing systems

**Enhanced Bottleneck Detection:**

- **Multi-Criteria Analysis**: Combines density, speed, queue length, and waiting time for robust bottleneck identification
- **Predictive Capability**: Identifies potential bottlenecks before they fully form using advanced traffic indicators
- **Update Frequency**: Configurable detection intervals (default: 60 seconds) independent of Tree Method timing
- **Enhanced Sensitivity**: More granular detection compared to Tree Method's speed-only analysis

**ATLCS Dynamic Pricing Engine:**

- **Congestion-Based Pricing**: Calculates priority scores based on traffic congestion severity using economic pricing theory
- **Signal Priority Translation**: Converts pricing data into traffic light extension recommendations
- **Real-Time Updates**: Rapid pricing recalculation (default: 5 seconds) for immediate traffic response
- **Priority Thresholds**: Three-tier priority system (high/medium/low) with corresponding extension durations

**Implementation Process:**

**Initialization Phase:**
- **Tree Method Foundation**: Inherits complete Tree Method infrastructure (Network, Graph, algorithm objects)
- **ATLCS Components**: Adds enhanced bottleneck detector, pricing engine, and demand-supply coordinator
- **Coordination Setup**: Initializes junction control state tracking for handoff management
- **Configuration Loading**: Applies bottleneck detection and ATLCS timing intervals from CLI arguments

**Runtime Behavior:**

**Coordinated Operation Cycle:**

1. **Tree Method Strategic Phase** (every 90 seconds):
   - Performs network-wide optimization calculations
   - Detects multi-edge trees indicating complex bottlenecks
   - Claims exclusive control of junctions with complex trees
   - Populates baseline phase durations in shared variables

2. **ATLCS Tactical Phase** (every 5-60 seconds):
   - Enhanced Bottleneck Detection: Identifies bottlenecks using multi-criteria analysis
   - ATLCS Pricing Calculation: Computes congestion-based priority scores
   - Junction Control Check: Verifies control permissions before modifications
   - Signal Extensions: Applies green phase extensions only to available junctions

3. **Per-Step Coordination**:
   - Tree Method updates traffic lights using shared durations (potentially modified by ATLCS)
   - ATLCS coordination prevents conflicts through jurisdiction boundaries
   - Control handoffs occur based on traffic conditions, not arbitrary timing

**Dynamic Pricing Implementation:**

- **Congestion Assessment**: Analyzes traffic indicators to calculate edge-specific pricing
- **Priority Classification**: 
  - High Priority (price > high threshold): Significant green time extension (8-12 seconds)
  - Medium Priority (price > medium threshold): Moderate extension (4-6 seconds)
  - Low Priority: No intervention, baseline Tree Method control maintained
- **Extension Logic**: Extends current green phases dynamically rather than recalculating entire cycle
- **Conflict Prevention**: Only modifies junctions not under Tree Method exclusive control

**Junction-Level Handoff Mechanism:**

- **Control States**: Each junction tracked as "tree_method", "atlcs", or "available"
- **Claim Logic**: Tree Method claims control when detecting multi-edge trees (complex bottlenecks requiring strategic optimization)
- **Release Logic**: Tree Method releases control when only single-edge trees remain (ATLCS can handle tactically)
- **ATLCS Compliance**: ATLCS respects Tree Method control ownership, skips modifications to claimed junctions

**Key Advantages:**

- **Conflict-Free Coordination**: Junction-level handoff eliminates method contradictions and computational waste
- **Complementary Specialization**: Strategic optimization (Tree Method) combined with tactical responsiveness (ATLCS)
- **Real-Time Adaptability**: Sub-10-second response times for bottleneck prevention while maintaining network optimization
- **Enhanced Detection**: Multi-criteria bottleneck identification superior to single-metric approaches
- **Economic Foundation**: Dynamic pricing theory provides robust priority calculation framework
- **Scalable Architecture**: Clear separation of concerns enables future enhancements and additional coordination methods

**Configuration Integration:**

- **Enhanced Bottleneck Detection Timing**: `--bottleneck-detection-interval` controls enhanced bottleneck detection frequency (default: 60 seconds)
- **ATLCS Timing**: `--atlcs-interval` controls dynamic pricing update frequency (default: 5 seconds)
- **Tree Method Integration**: Inherits `--tree-method-interval` for strategic calculation timing (default: 90 seconds)
- **Independent Operation**: All three timing systems operate independently for optimal performance balance

**Setup Requirements:**

- **Tree Method Foundation**: Complete Tree Method infrastructure including Network JSON, Graph objects, and cycle calculations
- **ATLCS Extensions**: Enhanced detector, pricing engine, and coordination components
- **Shared Variables**: Current phase durations dictionary for Tree Method-ATLCS communication
- **Control State Tracking**: Junction ownership management for handoff coordination

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