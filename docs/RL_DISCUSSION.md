# Reinforcement Learning for Network-Wide Traffic Control

This work explores the application of reinforcement learning to network-wide traffic signal control, with the objective of improving overall throughput and reducing vehicle waiting times. The general problem requires coordinated decision-making across multiple intersections and the ability to generalize to diverse congestion patterns. To make the problem tractable, we begin by examining key formulation choices and adopting a simplified configuration. This configuration translates the theoretical problem into a concrete reinforcement learning environment that can be systematically trained and evaluated. The resulting training process and evaluation framework provide a basis for assessing performance in this simplified setting and for motivating extensions toward more scalable and generalizable approaches.

## 1. Background: The General Problem

Traffic signal control is central to managing urban mobility, where poorly timed signals lead to congestion and longer travel times. Improving overall throughput and reducing waiting times requires coordinated decisions across multiple intersections, since congestion at one junction can propagate downstream.

Reinforcement learning offers a flexible framework for adaptive control under dynamic conditions, but applying it at the network scale introduces additional challenges. Policies must generalize across diverse congestion patterns and intersection configurations rather than memorizing behaviors for individual junctions. Without mechanisms that encourage such generalization, an RL agent risks treating each intersection and traffic pattern as unique, limiting scalability to larger networks.

This combination of domain-specific complexity and machine learning challenges defines the general problem of reinforcement learning for network-wide traffic signal control. Of particular importance are the reward structure choices, which fundamentally determine what coordination behaviors the agent learns and how effectively it can attribute network-wide performance to specific signal timing decisions. To address these challenges in a tractable way, we begin by examining key formulation choices and adopting a simplified configuration for initial experimentation.

## 2. Problem Formulation Choices

Designing an RL-based traffic signal controller requires a set of fundamental formulation choices. These decisions define the scope of learning, the structure of control, the information available, and the constraints under which the system operates:

- Training Scope – should the model be trained for a specific network or designed to generalize across networks?
- Agent Architecture – should control be centralized or distributed across junctions?
- Input Model – should the state representation capture traffic at a microscopic, macroscopic, hybrid, or graph-based level?
- Reward Design – should optimization be based on local intersection performance, global network objectives, or a hybrid of both?
- Action Representation – should the agent select phases, specify phase durations, or incrementally extend/terminate greens?
- Time Resolution – should decisions be made every simulation tick, at fixed intervals, or when signal constraints (e.g., min-green) expire?
- Exploration & Constraints – should training allow unconstrained exploration or enforce safety rules such as minimum green times and clearance intervals?

Each axis introduces trade-offs in complexity, scalability, realism, and performance.

### Training Scope

Traffic signal controllers must operate on diverse road networks, from small grid layouts to large irregular urban topologies. A fundamental choice is whether to train a model tailored to one specific network or to design a general controller that can transfer across different network configurations.

**Network-Specific Training**: The model is trained exclusively on a single, fixed network topology with customized state representations and architectures optimized for that specific configuration.

- Pros: Straightforward implementation due to fixed topology; can exploit known geometry and capacity constraints; often achieves higher performance on the target network; simplified model design and training process.
- Cons: Zero portability to other networks; requires complete retraining for even minor structural changes; limited research generalizability; cannot leverage learning from diverse network configurations.

**Network-Agnostic Training**: The model is designed to handle variable network sizes and connectivity patterns, typically using graph-based or attention-based architectures that can adapt to different topologies.

- Pros: Single controller works across multiple networks; enables transfer learning and fine-tuning; higher research impact through generalizability; can leverage diverse training scenarios to improve robustness.
- Cons: Increased engineering and computational complexity; may underperform compared to specialized models; requires sophisticated architectures to handle variable inputs; sacrifices some network-specific optimization capacity.

Network-specific training offers the fastest path to demonstrating effectiveness on a target network, making it ideal for proof-of-concept research or deployment in fixed infrastructure. Network-agnostic approaches require significantly more development effort but provide broader applicability. A practical compromise is to begin with network-specific training to validate core concepts, then extend to network-agnostic formulations once the approach proves viable.

### Agent Architecture

The control architecture determines how decision-making is organized across the network's intersections. This choice fundamentally affects scalability, coordination capability, and computational requirements.

**Centralized Architecture**: A single agent observes the entire network state and makes coordinated decisions for all intersections simultaneously.

- Pros: Direct optimization of network-wide coordination; can implement complex multi-intersection strategies like green waves; unified global reward optimization; no credit assignment problems across agents.
- Cons: State-action space grows exponentially with network size; computationally prohibitive for large networks; requires significant memory and processing power; single point of failure.

**Distributed Architecture**: Multiple agents operate independently, typically one per intersection, usually with shared parameters to enable learning transfer across similar junction configurations.

- Pros: Linear scaling with network size; tractable computation for large networks; natural parallelization; robust to individual junction failures; aligns with real-world distributed control systems.
- Cons: Coordination must emerge indirectly through learning; difficult credit assignment for multi-junction improvements; potential training instability from parameter sharing; may miss optimal network-wide strategies.

Centralized control is optimal for small research networks where full coordination is critical and computational constraints are manageable. Distributed control becomes necessary for larger, real-world networks due to scalability requirements. A hybrid approach uses centralized training with distributed execution, allowing coordination learning while maintaining deployment scalability.

### Input Model

The input representation defines how traffic conditions are observed and encoded for the RL agent. This choice determines the level of detail available for decision-making, computational requirements, and the agent's ability to generalize across different traffic patterns.

**Microscopic Representation**: Individual vehicle states including positions, speeds, lane assignments, and trajectories for every vehicle in the network.

- Pros: Complete system detail enabling precise trajectory prediction; captures fine-grained vehicle interactions; allows modeling of individual driver behaviors; maximum information availability.
- Cons: Enormous state space scaling poorly with traffic volume; location-specific features hindering transferability; slow training due to high dimensionality; requires specialized hardware for real-time inference.

**Macroscopic Representation**: Aggregate statistics per road segment such as queue lengths, average speeds, traffic densities, and flow rates without tracking individual vehicles.

- Pros: Compact and stable representation; matches traffic control granularity; efficient for training and real-time deployment; highlights network-level patterns; enables standard hardware deployment.
- Cons: Loses individual vehicle dynamics; misses fine-grained intersection behaviors; may obscure important microscopic effects; reduced precision for complex maneuvers.

**Multi-Scale Hybrid**: Dynamic combination of microscopic detail at critical locations (bottlenecks, complex intersections) with macroscopic representation elsewhere based on current traffic conditions.

- Pros: Adaptive detail allocation optimizing information-efficiency trade-off; balances computational cost with necessary precision; can focus resources on problem areas.
- Cons: Complex architecture requiring dynamic switching logic; potential instability when transitioning between representations; difficult to implement and debug; inconsistent state space.

**Graph-Based Representation**: Network topology encoded explicitly as a graph with junction nodes (signal states, flow features) and road edges (density, congestion, capacity), typically processed using Graph Neural Networks.

- Pros: Direct topology encoding enabling parameter sharing; natural handling of irregular networks; supports transfer to new configurations; captures local and global propagation through message passing.
- Cons: Requires careful feature engineering; computationally heavier than simpler approaches; limited maturity of GNN architectures for sequential decision-making; complex debugging due to message passing dynamics.

Macroscopic representation may provide the best balance of information content and computational efficiency for most traffic control applications. Microscopic detail is rarely necessary since traffic signals operate at aggregate flow levels, while graph-based approaches add complexity that may not justify performance gains for network-specific applications. Multi-scale hybrid approaches should be reserved for scenarios where computational resources are abundant and bottleneck locations are well-identified.

### Reward Design

The reward structure fundamentally determines what behaviors the agent learns and directly impacts training efficiency, credit attribution, and performance outcomes. Traffic control presents unique challenges where individual signal decisions have delayed effects, vehicle journeys span multiple intersections, and network performance depends on complex coordination patterns.

**Individual Vehicle Rewards**: Track each vehicle's journey through the network, providing completion bonuses and journey-specific waiting time penalties based on individual travel experiences.

- Pros: Natural load balancing across routes and traffic patterns; precise temporal attribution of performance improvements; encourages fair treatment of all vehicles; clear causality between decisions and outcomes.
- Cons: High computational overhead tracking individual vehicles; complex implementation requiring vehicle journey management; potential reward sparsity for long journeys; memory intensive for large traffic volumes.

**Statistical Vehicle Rewards**: Aggregate performance across all vehicles into network-wide metrics such as total throughput, average waiting time, queue lengths, or completion rates.

- Pros: Simple implementation using readily available traffic statistics; lower computational requirements; stable signals less affected by individual vehicle variance; aligns with traditional traffic engineering metrics.
- Cons: Obscures which specific decisions contributed to improvements; high variance and attribution challenges; may miss important distributional effects; delayed feedback reducing learning efficiency.

**Global Spatial Scope**: Rewards computed across the entire network, encouraging system-wide optimization and coordination between all intersections.

- Pros: Captures network-wide coordination effects; encourages green wave progression and bottleneck relief; aligns with true optimization objectives; prevents local optimization at system expense.
- Cons: High reward variance making learning unstable; difficult credit assignment to individual decisions; all agents receive identical rewards regardless of contribution; may mask important local inefficiencies.

**Local Spatial Scope**: Rewards computed separately for each intersection or local region, focusing on junction-specific performance metrics.

- Pros: Clear attribution to specific junction decisions; lower variance enabling stable learning; easier debugging and interpretation; natural for distributed agent architectures.
- Cons: Fails to capture inter-junction coordination; may encourage selfish behavior harming network performance; misses downstream effects of local decisions; ignores global optimization objectives.

**Episode-Based Rewards**: Final performance evaluation provided only at episode completion, typically measuring total travel time or throughput over the entire simulation period.

- Pros: Captures long-term performance effects; aligns with ultimate optimization objectives; simple implementation; reduces computational overhead during training.
- Cons: Sparse learning signals leading to slow convergence; high gradient variance from delayed feedback; difficulty identifying which decisions contributed to outcomes; poor sample efficiency.

**Intermediate Rewards**: Periodic network performance measurements provided during episodes at regular intervals or triggered by specific events.

- Pros: More frequent feedback reducing learning variance; enables identification of effective decision sequences; better sample efficiency; allows correction of poor strategies mid-episode.
- Cons: Requires careful frequency tuning to avoid noise; may encourage short-term optimization; computational overhead from frequent measurements; potential interference with long-term learning.

A hybrid approach combining individual vehicle tracking with intermediate global rewards may provide the best balance. Individual vehicle rewards enable precise credit assignment while intermediate global measurements capture coordination effects. The reward frequency should match the time scale of traffic signal effects to balance learning efficiency with computational practicality. Local rewards should be avoided unless computational constraints require distributed learning with minimal communication.

### Action Representation

The action space defines how the RL agent controls traffic signal operations. This choice determines the granularity of control, learning complexity, and alignment with real-world signal systems.

**Phase Selection**: The agent selects which traffic signal phase (predefined movement pattern) should be active, with phase durations determined by fixed timing plans or separate mechanisms.

- Pros: Simple discrete action space enabling stable learning; aligns with standard traffic engineering practice; reduces action space dimensionality; prevents invalid signal combinations; easy integration with existing signal infrastructure.
- Cons: Limited flexibility in timing optimization; cannot adapt phase durations to traffic conditions; may miss optimal timing strategies; requires separate mechanism for duration control.

**Phase + Duration**: The agent simultaneously selects both the active phase and its green time duration, providing complete control over signal timing.

- Pros: Maximum flexibility enabling optimal timing adaptation; single decision captures complete signal strategy; can learn complex timing patterns; direct optimization of phase durations.
- Cons: Larger action space complicating learning; potential for unsafe or unrealistic timing choices; increased training instability; requires careful constraint enforcement.

**Incremental Control**: The agent makes binary decisions to extend the current phase or terminate it and advance to the next phase in the sequence.

- Pros: Most realistic reflecting actual signal controller operations; stable learning due to constrained action space; natural enforcement of minimum/maximum timing constraints; enables adaptive timing without complex duration prediction.
- Cons: Limited ability to skip phases or implement non-sequential strategies; may require multiple decisions to achieve desired timing; less direct control over final phase durations.

Phase selection with fixed durations may provide the most stable learning environment and fastest convergence, making it ideal for proof-of-concept research and initial validation. Incremental control offers the best balance of realism and learnability for practical deployment, as it mirrors how actual traffic controllers operate. Phase + duration control should be reserved for scenarios where timing flexibility is critical and sufficient training time is available to handle the increased complexity.

### Time Resolution

The decision frequency determines how often the RL agent makes control decisions. This choice affects policy stability, computational load, responsiveness to traffic changes, and alignment with real-world signal operations.

**High Frequency (Every Simulation Tick)**: The agent makes decisions at every simulation time step, typically every 1-2 seconds, providing maximum temporal granularity.

- Pros: Maximum responsiveness to traffic changes; finest control granularity; can react immediately to emerging conditions; captures all temporal dynamics.
- Cons: Produces noisy and unstable policies; high computational overhead; unrealistic for real-world deployment; increases sample complexity; may lead to erratic signal behavior.

**Fixed Intervals**: The agent makes decisions at regular time intervals, typically every 5-10 seconds, independent of current signal state or traffic conditions.

- Pros: Smooth and stable decision-making; computationally efficient; aligns well with traffic signal operations; reduces policy noise; predictable computational load.
- Cons: May miss critical timing opportunities; less responsive to rapid traffic changes; arbitrary timing intervals may not match traffic dynamics; potential inefficiency during low-traffic periods.

**Event-Driven**: The agent makes decisions triggered by specific events such as minimum green time expiration, maximum green time approach, or significant traffic condition changes.

- Pros: Most efficient use of decisions; adaptive timing matching traffic needs; natural alignment with signal constraints; reduces unnecessary decisions during stable periods; realistic operational model.
- Cons: Variable computational load; requires complex state management; may miss gradual traffic changes; implementation complexity; unpredictable decision timing.

Fixed interval decision-making at 5-10 second intervals may provide the optimal balance for most applications, offering stable learning while maintaining reasonable responsiveness. Event-driven approaches are most realistic and efficient but require sophisticated implementation. High-frequency decisions should be avoided except in research scenarios specifically studying fine-grained temporal control, as they introduce instability without corresponding performance benefits in traffic control applications.

### Exploration & Constraints

The exploration strategy determines how the RL agent discovers effective policies while respecting safety and operational constraints. This choice affects training safety, sample efficiency, and the realism of learned behaviors.

**Constrained Exploration**: The agent's action selection is restricted to feasible signal sequences that respect minimum green times, yellow clearance intervals, maximum green durations, and phase ordering constraints.

- Pros: Ensures all explored policies are safe and realistic; prevents wasted samples on infeasible actions; simplifies training by eliminating invalid states; aligns with real-world operational requirements; reduces need for post-hoc constraint enforcement.
- Cons: May limit discovery of novel strategies; restricts exploration space potentially missing optimal solutions; requires careful constraint implementation; may overconstrain and block useful timing strategies.

**Unconstrained Exploration**: The agent can select any action without operational restrictions, maximizing exploration flexibility and policy discovery potential.

- Pros: Maximum flexibility for discovering innovative strategies; no artificial limitations on policy search; simpler implementation without constraint logic; may find unexpected optimal solutions; enables learning of constraint importance through experience.
- Cons: Risk of unsafe or unrealistic signal behaviors during training; wasted samples on infeasible actions; potential for learning policies that cannot be deployed; may require extensive post-training validation; possible safety violations in simulation.

**Penalty-Based Constraints**: The agent can select any action but receives negative rewards for violating operational constraints, learning to avoid unsafe behaviors through experience.

- Pros: Balances exploration flexibility with constraint learning; enables discovery of constraint boundaries; natural integration of safety requirements into learning; allows gradual constraint tightening; maintains exploration while encouraging feasible behaviors.
- Cons: May waste significant training time learning basic constraints; requires careful penalty tuning; potential for persistent constraint violations; slower convergence compared to hard constraints; may not guarantee constraint satisfaction.

Constrained exploration provides the most practical approach for traffic signal control, ensuring all learned behaviors are deployable while maintaining sufficient exploration for effective learning. The constraints should match real-world signal requirements including minimum green times (typically 5-10 seconds), appropriate clearance intervals, and maximum green durations (60-120 seconds). Unconstrained exploration should only be used in research contexts where understanding constraint effects is specifically of interest, as it significantly reduces training efficiency and deployment viability.

### Summary

These formulation choices are deeply interdependent. The training scope shapes the viability of the input model: network-agnostic approaches require graph-based or aggregated features, while network-specific models can rely on simpler encodings. The agent architecture interacts with reward design, since distributed agents need local or hybrid rewards for effective learning, whereas centralized control can optimize a single global objective. Likewise, the action representation and time resolution must align with real-world signal operations, ensuring that decisions are both learnable and realistic. Finally, exploration and constraints influence all other axes, since the boundaries imposed during training determine which strategies the agent can discover. In practice, effective RL controllers emerge not from optimizing each choice in isolation, but from carefully balancing these dimensions to fit the deployment context.

**Recommended Starting Configuration**:

- Training Scope: Network-specific (much faster training, sufficient for proof-of-concept)
- Agent Architecture: Centralized (direct coordination, optimal for proof-of-concept)
- Input Model: Macroscopic (compact state space, stable training)
- Reward Design: Individual vehicle rewards, global spatial scope, intermediate rewards, delta measurement, time-windowed credit distribution
- Action Representation: Phase + Discrete Duration Selection (simplified discrete action space)
- Time Resolution: Flexible parameter for empirical optimization (balance responsiveness vs stability)
- Exploration & Constraints: Enforce minimum green times (realistic, prevents unsafe policies)

## 3. RL Environment Design

This chapter translates the design choices from Chapter 2 into a concrete RL environment architecture. We focus on the recommended starting configuration: centralized agent with macroscopic state representation, global reward design, and phase+duration control. The goal is to establish the architectural foundation that bridges theoretical formulation with practical implementation, ensuring the RL system can effectively learn network-wide traffic coordination.

### Environment Architecture Overview

The RL environment acts as a wrapper around the existing SUMO simulation infrastructure, providing a standard OpenAI Gym interface while leveraging the sophisticated traffic analysis capabilities already built into the Tree Method system. This architectural choice enables direct performance comparison between RL and Tree Method approaches using identical traffic state representations and simulation conditions.

**System Integration Strategy**: Rather than rebuilding traffic simulation capabilities, the RL environment extends the existing infrastructure. The Tree Method already computes the macroscopic traffic indicators needed for RL state representation—speeds, densities, flow rates, and congestion status. The RL system reuses these calculations while adding the vehicle tracking and reward computation components necessary for learning. This integration approach ensures consistency between baseline and RL evaluations while minimizing implementation complexity.

### State Space Design

The state representation implements the macroscopic input model from Chapter 2, providing the centralized agent with comprehensive network-wide traffic information in a compact, normalized format suitable for neural network processing.

**Traffic Flow Representation**: Each network edge contributes four normalized traffic indicators that capture current conditions and congestion status. Speed values represent current flow efficiency relative to free-flow conditions. Density measurements using traffic flow theory indicate edge utilization levels. Flow rates capture actual vehicle throughput on each edge. Congestion flags provide binary indicators of bottleneck conditions that require immediate attention.

**Signal State Representation**: Each intersection contributes timing and phase information that enables coordinated signal control. Current active phases indicate the present signal configuration across all network intersections. Phase timing information shows remaining durations and recent transition history. This signal context allows the centralized agent to make informed coordination decisions based on current network-wide signal states.

**Network Connectivity Context**: The state representation includes spatial relationships between network elements to support coordination learning. Upstream and downstream connectivity information enables the agent to understand how local decisions affect neighboring intersections. Traffic propagation patterns from congested areas help identify coordination opportunities for implementing strategies like green wave progression.

**State Vector Construction**: All network information is concatenated into a single normalized vector with fixed dimensionality determined by the network topology. For a network with E edges and J junctions, the state vector has dimensionality E × 4 + J × 2 (four traffic indicators per edge plus two signal features per junction). This fixed-size representation enables efficient neural network processing while capturing all essential traffic and signal information needed for coordination decisions.

### Action Space Design

The action space implements the phase+duration control choice from Chapter 2, enabling the centralized agent to make coordinated timing decisions across all network intersections simultaneously.

**Fully Discrete Actions**: Each intersection requires two discrete control decisions: which traffic signal phase to activate and which duration to select from predefined options (e.g., 10, 15, 20, 30, 45, 60, 90, 120 seconds). This discrete action space simplifies learning while maintaining practical control flexibility and alignment with real-world signal operations.

**Centralized Coordination**: The centralized agent simultaneously controls all intersections in the fixed network topology through discrete action selection. This enables direct coordination across the entire network while providing stable learning through the simplified discrete action space.

**Safety Constraint Integration**: Duration actions are bounded by traffic engineering safety requirements including minimum green times and maximum phase durations. These constraints prevent the agent from learning unsafe or unrealistic signal behaviors while maintaining sufficient exploration space for effective timing optimization. The constraint enforcement becomes part of the environment interface, ensuring all learned policies remain deployable in real-world systems.

### Reward System

The reward system uses two separate components: intermediate vehicle penalties during episodes and throughput bonuses at episode completion.

**Intermediate Vehicle Penalties**: At regular measurement intervals, vehicles with increased waiting time generate penalties applied to global network decisions:

```
For each vehicle i at measurement time:
  Δwait_time_i = current_waiting_time_i - previous_waiting_time_i
  if Δwait_time_i > 0:
    penalty_i = -Δwait_time_i
    Apply penalty_i to ALL global decisions made since vehicle_i started
```

**Global Decision Attribution**: Each vehicle's penalty affects all network-wide signal control decisions made during that vehicle's journey. Multiple vehicles can penalize the same global decision if they were all in the network when it was made.

**Episode Completion Rewards**: Only at episode end, provide throughput bonuses:

```
R_episode = α × total_completed_vehicles
```

The scaling factor α balances episode rewards against accumulated intermediate penalties. If α is too low, the agent focuses primarily on reducing individual vehicle waiting times but may sacrifice overall throughput. If α is too high, the agent prioritizes completing vehicles quickly but may ignore service quality for vehicles currently in the network. The optimal α value depends on network characteristics (size, congestion levels) and training objectives.

## 4. Training Framework Strategy

This chapter establishes the strategic approach for training RL agents on network-wide traffic control. We focus on algorithm selection rationale, training configuration principles, and the integration of vehicle-based credit assignment with modern RL algorithms. The framework builds on Chapter 3's environment design to create an effective learning system for coordinated signal control.

### Algorithm Selection Rationale

Selecting the right RL algorithm for traffic control requires careful consideration of domain-specific constraints that distinguish this application from standard RL problems.

**Domain-Specific Algorithm Requirements**: Traffic signal control imposes unique constraints on algorithm choice. The fully discrete action space (phase selection + discrete duration selection) enables standard RL algorithms including both value-based and policy gradient methods. Expensive simulation episodes demand high sample efficiency, favoring algorithms with proven performance on discrete action spaces. The centralized coordination requirement favors methods that can handle coordinated multi-intersection control effectively.

**Algorithm Categories and Traffic Control Fit**:

**Value-Based Methods**: DQN can handle discrete phase and duration selection through standard Q-value estimation for each intersection. The discrete action space enables efficient exploration and often provides better sample efficiency than policy gradient methods in traffic control scenarios.

**Policy Gradient Methods**: Direct policy optimization may provide better coordination learning through joint action optimization. These methods can handle the discrete action space naturally while potentially discovering better coordination patterns than value-based approaches.

**Actor-Critic Hybrid Approaches**: Combining policy gradients with value function guidance provides the benefits of direct policy optimization while adding variance reduction through learned value estimates. The actor-critic architecture naturally aligns with traffic control intuition: the critic evaluates traffic states while the actor decides signal timing actions.

**Algorithm Options**: Both DQN variants and PPO become viable choices for centralized traffic control. DQN offers simpler implementation and often better sample efficiency for discrete action spaces, while PPO may provide superior coordination learning through joint action optimization. The choice between them depends on training preferences: DQN for simplicity and sample efficiency, or PPO for potentially better coordination discovery.

### Training Configuration Strategy

Effective training for traffic control requires domain-specific configuration that accounts for expensive simulations, complex coordination requirements, and the unique reward structure established in Chapter 3.

**Training Stability Considerations**: Traffic control training faces several stability challenges that require careful algorithmic configuration. The large action space from centralized control can lead to training instability if policy updates are too aggressive. The two-component reward system creates varying gradient magnitudes that need proper balancing. The complex coordination requirements mean that small policy changes can have large performance effects, requiring conservative training approaches.

**Sample Efficiency Requirements**: Traffic simulations are computationally expensive, making sample efficiency a critical constraint. Training configurations must maximize learning from limited environment interactions. This requirement favors algorithms with good sample efficiency and training approaches that can leverage existing domain knowledge rather than learning everything from scratch.

**Hyperparameter Strategy for Traffic Control**: Domain-specific hyperparameter choices can significantly improve training effectiveness. Conservative learning rates prevent destabilization of learned coordination patterns. Appropriate batch sizes balance learning stability with memory constraints from large state vectors. Clip ranges should be tuned to prevent large policy updates while maintaining sufficient exploration capacity.

### Credit Assignment Integration

The vehicle-based reward system from Chapter 3 requires specific integration with the chosen RL algorithm to ensure stable learning and effective credit attribution.

**Multi-Signal Reward Handling**: Unlike standard RL applications with single reward signals, the traffic control system generates multiple simultaneous vehicle penalties at each time step. The training framework must aggregate these signals appropriately while preserving the learning information from individual vehicle experiences.

**Temporal Credit Distribution**: The time-windowed credit assignment strategy connects vehicle performance outcomes to the specific signal decisions that influenced those outcomes. This temporal attribution mechanism must integrate with the algorithm's gradient computation to ensure recent decisions receive appropriate credit for their effects on vehicle performance.

**Variance Management**: The frequent intermediate rewards from vehicle tracking can increase gradient variance compared to sparse episode-based signals. The training framework must balance the dense feedback benefits against potential training instability from high-variance gradients.

**Reward Component Balancing**: The dual reward structure requires careful weighting between immediate vehicle penalties and episode throughput rewards. The training framework must provide mechanisms for tuning this balance to ensure the agent learns both responsive vehicle service and effective network-wide coordination.

## 5. Implementation Guide

This chapter provides detailed step-by-step instructions for building the RL traffic control system. We'll implement a custom OpenAI Gym environment that integrates with the existing SUMO/Tree Method infrastructure, followed by PPO training using Stable-Baselines3.

### Implementation Architecture Overview

Before diving into specific implementation steps, it's important to understand how all the RL components fit together with the existing system.

**System Architecture**: The RL implementation consists of four main components that work together: a Gym environment wrapper that interfaces with SUMO simulation, a vehicle tracking system that monitors individual vehicle journeys for reward computation, a state collection system that gathers network-wide traffic information, and a training pipeline that orchestrates the learning process using PPO.

**Integration with Existing Infrastructure**: The RL system extends rather than replaces the existing Tree Method infrastructure. It reuses the sophisticated traffic analysis calculations already implemented while adding vehicle tracking and reward computation capabilities. This approach ensures consistency between RL and Tree Method evaluations while minimizing implementation complexity.

**Data Flow Design**: Traffic information flows from SUMO through the existing analysis pipeline to create RL state vectors. Vehicle performance data flows through the new tracking system to generate reward signals. Control actions flow from the RL agent through the existing signal control interface back to SUMO. This bidirectional integration maintains all existing capabilities while enabling RL learning.

### Implementation Roadmap

The implementation follows a systematic build-and-test approach that ensures each component works correctly before adding complexity.

**Phase 1: Environment Skeleton** - Create basic Gym environment structure with dummy implementations to verify interface compatibility with RL libraries before adding complexity.

**Phase 2: State Integration** - Connect the environment to existing traffic analysis calculations and implement state vector construction and normalization.

**Phase 3: Action Execution** - Implement action processing and integrate with existing signal control interface, including safety constraint enforcement.

**Phase 4: Vehicle Tracking** - Add individual vehicle monitoring and basic reward computation capabilities.

**Phase 5: Complete Reward System** - Implement full two-component reward architecture with temporal credit assignment.

**Phase 6: Training Integration** - Create PPO training script and validate end-to-end learning functionality.

**Testing Strategy**: Each phase includes component-specific tests using simple 3×3 grid networks before scaling to larger configurations. Validation compares RL environment outputs with Tree Method outputs to ensure consistency.

### Required Dependencies and Setup

**Library Requirements**: Install Stable-Baselines3 for PPO implementation, OpenAI Gym for environment interface, and NumPy for numerical operations. Ensure existing SUMO/TraCI dependencies remain functional.

**Development Environment**: The implementation builds on the existing Python codebase structure. Create new files in `src/rl/` directory while extending existing classes in `src/orchestration/` and `src/sumo_integration/` as needed.

### Step 1: Gym Environment Foundation

**TrafficControlEnv Class Structure**: Create a new class inheriting from `gym.Env` that wraps the existing SUMO simulation infrastructure. Initialize the environment with the same configuration system used by Tree Method, ensuring identical network and traffic conditions for fair comparison.

```python
class TrafficControlEnv(gym.Env):
    def __init__(self, config):
        # Initialize with existing configuration system
        # Load same Graph and JunctionNode objects as Tree Method
        # Set up observation and action spaces

    def step(self, action):
        # Process RL actions and apply to SUMO
        # Advance simulation one time step
        # Collect state observations
        # Compute rewards from vehicle tracking
        # Return (observation, reward, done, info)

    def reset(self):
        # Restart SUMO simulation
        # Initialize vehicle tracking
        # Return initial observation
```

**Observation Space Design**: Define the state space using `gym.spaces.Box` with normalized values between 0.0 and 1.0. Calculate dimensions from network topology: `num_edges × edge_features + num_intersections × intersection_features`. Each edge contributes traffic flow indicators while each intersection contributes signal timing information.

**Action Space Configuration**: Implement continuous action space using `gym.spaces.Box` where each intersection requires phase selection values and duration values. The action space dimension equals `2 × num_intersections`. Process actions through softmax for phase selection and scaling for duration values within safety bounds.

**Environment Lifecycle Management**: Implement the core Gym interface methods with proper integration to existing simulation infrastructure. The `step()` method coordinates action execution, simulation advancement, state collection, and reward computation. The `reset()` method handles simulation restart and state initialization.

### Step 2: Vehicle Tracking System

**VehicleTracker Class Implementation**: Create a dedicated class to monitor individual vehicle journeys and compute reward signals based on performance changes.

```python
class VehicleTracker:
    def __init__(self):
        self.vehicle_histories = {}  # Track journey data per vehicle
        self.decision_timestamps = []  # Log signal control decisions
        self.last_measurement_time = 0

    def update_vehicles(self, current_time):
        # Get active vehicles from TraCI
        # Update waiting times and journey data
        # Compute penalties for vehicles with increased waiting times

    def record_decision(self, timestamp, actions):
        # Log signal timing decisions with timestamps
        # Enable credit assignment to recent decisions

    def compute_intermediate_rewards(self):
        # Calculate vehicle-based penalties
        # Apply time-windowed credit distribution
        # Return aggregated reward signal
```

**Journey Tracking Design**: Monitor each vehicle from network entry to completion, maintaining records of start times, routes, and accumulated waiting times. Handle dynamic vehicle populations as vehicles enter and leave the simulation throughout episodes.

**Reward Computation Strategy**: Implement the two-component reward system with intermediate vehicle penalties and final episode throughput bonuses. Calculate waiting time deltas at measurement intervals and convert to penalty signals. Provide episode completion rewards based on total vehicle throughput.

**Credit Assignment Implementation**: Build the time-windowed attribution system that connects vehicle performance outcomes to relevant signal control decisions. Maintain decision timestamp records and apply penalties to decisions made since each vehicle's journey began.

### Step 3: State Collection Integration

**State Vector Construction**: Build the observation vector by integrating with existing traffic analysis calculations. Reuse the sophisticated traffic flow computations already implemented in the Tree Method system rather than reimplementing traffic analysis.

```python
def build_state_vector(self):
    edge_features = []
    for edge in self.network.edges:
        # Collect normalized traffic indicators
        speed_norm = edge.current_speed / edge.speed_limit
        density_norm = edge.density / MAX_DENSITY
        flow_norm = edge.flow / edge.capacity
        congestion_flag = 1.0 if edge.is_congested else 0.0
        edge_features.extend([speed_norm, density_norm, flow_norm, congestion_flag])

    junction_features = []
    for junction in self.network.junctions:
        phase_norm = junction.current_phase / len(junction.phases)
        duration_norm = junction.remaining_time / MAX_PHASE_TIME
        junction_features.extend([phase_norm, duration_norm])

    return np.array(edge_features + junction_features)
```

**Normalization Strategy**: Ensure all state components remain within [0.0, 1.0] bounds using network-specific parameters. Use speed limits, capacity constraints, and timing bounds for consistent normalization across episodes.

### Step 4: Action Execution System

**Action Processing**: Convert RL agent outputs into valid signal control commands while enforcing safety constraints.

```python
def process_actions(self, raw_actions):
    actions = []
    for i, junction in enumerate(self.network.junctions):
        # Extract phase and duration for this intersection
        phase_logits = raw_actions[i*2:(i*2)+len(junction.phases)]
        duration_raw = raw_actions[i*2+1]

        # Process phase selection through softmax
        phase_probs = softmax(phase_logits)
        selected_phase = np.argmax(phase_probs)

        # Scale and constrain duration
        duration = np.clip(duration_raw * MAX_PHASE_TIME, MIN_PHASE_TIME, MAX_PHASE_TIME)

        actions.append((selected_phase, duration))
    return actions
```

**Signal Control Integration**: Execute processed actions through the existing signal control interface, ensuring compatibility with the current TraCI integration and maintaining safety constraint enforcement.

### Step 5: Training Script Development

**PPO Configuration**: Set up Stable-Baselines3 PPO with traffic control-specific parameters.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Validate environment implementation
env = TrafficControlEnv(config)
check_env(env)

# Configure PPO for traffic control
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=2e-4,  # Conservative for expensive episodes
    clip_range=0.1,      # Prevent large policy updates
    batch_size=1024,     # Balance stability and memory
    verbose=1
)

# Training execution
model.learn(total_timesteps=100000)
model.save("traffic_control_policy")
```

**Training Pipeline**: Implement monitoring and logging to track training progress, episode rewards, and convergence metrics. Include capabilities for training interruption and resumption.

### Step 6: Component Testing and Validation

**Incremental Testing Strategy**: Test each component independently before integration. Start with simple 3×3 grid networks and scale up to validate performance and stability.

**Validation Methodology**: Compare RL environment state outputs with Tree Method outputs to ensure consistency. Verify action execution produces valid signal sequences. Test reward computation across different traffic scenarios.

**Integration Testing**: Validate end-to-end functionality from environment initialization through training execution. Ensure the complete system can train effectively and produce viable policies.

## 6. Evaluation & Validation Methodology

This chapter provides comprehensive guidance for evaluating RL traffic control systems and validating implementation correctness. We cover experimental design principles, validation strategies, performance metrics, and practical debugging approaches to ensure reliable assessment of RL agent performance against baseline methods.

### Evaluation Strategy Framework

Effective evaluation of RL traffic control requires systematic comparison against established baselines using rigorous experimental methodology that accounts for the stochastic nature of both traffic simulation and RL training.

**Baseline Selection Strategy**: Compare RL agents primarily against Tree Method as the main baseline, since both systems use identical traffic state representations and simulation infrastructure. Include additional baselines like SUMO's actuated control and fixed timing for broader context. Ensure all methods operate on identical network configurations and traffic scenarios to eliminate confounding variables.

**Experimental Design Principles**: Use controlled experimentation with systematic parameter variation to assess RL performance across different traffic conditions. Test multiple network sizes, vehicle densities, and traffic patterns to evaluate generalization capability. Employ statistical methodology with multiple independent runs using different random seeds to account for both RL training variance and traffic simulation stochasticity.

**Fair Comparison Requirements**: Ensure all methods receive identical inputs including network topology, vehicle routes, departure patterns, and simulation parameters. Use the same SUMO configuration and TraCI integration for all approaches. Maintain consistent measurement intervals and evaluation metrics across different control methods to enable valid performance comparisons.

### Implementation Validation

Before conducting performance evaluations, thoroughly validate that the RL implementation works correctly and produces reasonable behaviors.

**Environment Validation**: Test the Gym environment implementation using Stable-Baselines3's built-in validation tools. Verify that state vectors maintain expected dimensionality and normalization ranges across different network configurations. Confirm that action processing produces valid signal timing sequences that respect safety constraints. Validate that reward computation generates reasonable signals and that episode management handles simulation reset correctly.

**State Collection Validation**: Compare RL environment state vectors with Tree Method state calculations to ensure consistency. The two systems should produce identical traffic flow measurements for the same simulation conditions. Test state collection across different traffic scenarios to verify robustness and accuracy.

**Action Execution Validation**: Verify that RL actions translate correctly into SUMO signal control commands. Test that phase selections and duration values remain within specified bounds. Confirm that signal transitions follow safe timing sequences and that multiple intersection coordination works correctly.

**Vehicle Tracking Validation**: Test the reward system with known traffic scenarios where outcomes can be predicted. Verify that individual vehicle tracking accurately captures journey information and waiting time changes. Validate that credit assignment correctly attributes performance outcomes to relevant signal decisions.

### Performance Evaluation Methodology

**Experimental Configuration**: Design experiments that systematically test RL performance across relevant traffic control scenarios while maintaining statistical rigor.

**Traffic Scenario Design**: Create diverse traffic scenarios that test different aspects of coordination capability including rush hour patterns with high demand, light traffic scenarios for efficiency testing, congestion bottleneck situations for coordination assessment, and mixed traffic patterns combining different vehicle types and routing strategies.

**Statistical Methodology**: Conduct multiple independent runs (minimum 20 per condition) with different random seeds for both RL training and traffic simulation. Use appropriate statistical tests to assess significance of performance differences. Report confidence intervals and effect sizes rather than just mean comparisons.

**Metrics and Measurement**: Focus on metrics that capture both system efficiency and individual vehicle service quality.

**System-Level Metrics**:

- **Total Throughput**: Number of vehicles completing their journeys successfully
- **Network Capacity Utilization**: Ratio of completed vehicles to total spawned vehicles
- **Average Network Travel Time**: Mean time from vehicle entry to exit across all completed journeys
- **Congestion Duration**: Total time network spends in congested states

**Individual-Level Metrics**:

- **Average Vehicle Waiting Time**: Mean waiting time per vehicle across all completed journeys
- **Maximum Individual Waiting Time**: Worst-case individual vehicle experience
- **Service Quality Distribution**: Percentile analysis of individual waiting times (50th, 90th, 95th percentiles)
- **Journey Completion Rate**: Percentage of vehicles reaching their destinations within episode time limits

### Success Criteria and Interpretation

**Performance Threshold Selection**: Choose success criteria based on domain knowledge and baseline performance rather than arbitrary percentages. Consider the magnitude of improvement demonstrated by Tree Method in research literature and set realistic expectations for RL performance.

**Recommended Success Criteria**:

- **Throughput Improvement**: RL achieves 5-15% improvement in completed vehicles vs Tree Method (accounting for training variance)
- **Waiting Time Reduction**: RL reduces average waiting time by 10-25% compared to baseline methods
- **Service Quality**: RL improves 90th percentile waiting times, demonstrating better worst-case performance
- **Consistency**: RL performance remains stable across different random seeds and traffic scenarios

**Interpretation Guidelines**: Assess not just mean performance improvements but also consistency and robustness across different conditions. Consider training stability and convergence characteristics as indicators of implementation quality. Analyze failure modes and edge cases to understand RL agent limitations and areas for improvement.

### Debugging and Troubleshooting

**Common Implementation Issues**:

**Training Instability**: If RL training shows high variance or poor convergence, check reward scaling balance between vehicle penalties and throughput rewards. Verify state normalization consistency and action space constraint enforcement. Consider reducing learning rates or increasing batch sizes for more stable updates.

**Poor Performance vs Baselines**: If RL underperforms baselines significantly, validate that state collection captures relevant traffic information. Check that action execution correctly translates to signal control. Verify reward system provides appropriate learning signals and not conflicting objectives.

**Integration Problems**: If the RL system produces errors or crashes, systematically test each component independently. Validate environment reset functionality and episode management. Check TraCI integration and simulation lifecycle handling.

**Debugging Methodology**: Use systematic diagnosis starting with component-level validation before investigating system-level issues. Log intermediate values for state vectors, actions, and rewards to identify problematic patterns. Test with progressively complex scenarios starting from simple traffic conditions.

**Performance Analysis Tools**: Implement logging and visualization tools to analyze RL agent behavior during training and evaluation. Track learning curves, action distributions, and reward patterns to identify optimization opportunities and diagnose training issues.
