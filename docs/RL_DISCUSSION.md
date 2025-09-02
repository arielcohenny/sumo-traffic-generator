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

Macroscopic representation provides the best balance of information content and computational efficiency for most traffic control applications. Microscopic detail is rarely necessary since traffic signals operate at aggregate flow levels, while graph-based approaches add complexity that may not justify performance gains for network-specific applications. Multi-scale hybrid approaches should be reserved for scenarios where computational resources are abundant and bottleneck locations are well-identified.

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

A hybrid approach combining individual vehicle tracking with intermediate global rewards often provides the best balance. Individual vehicle rewards enable precise credit assignment while intermediate global measurements capture coordination effects. The reward frequency should match the time scale of traffic signal effects to balance learning efficiency with computational practicality. Local rewards should be avoided unless computational constraints require distributed learning with minimal communication.

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

Phase selection with fixed durations provides the most stable learning environment and fastest convergence, making it ideal for proof-of-concept research and initial validation. Incremental control offers the best balance of realism and learnability for practical deployment, as it mirrors how actual traffic controllers operate. Phase + duration control should be reserved for scenarios where timing flexibility is critical and sufficient training time is available to handle the increased complexity.

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

Fixed interval decision-making at 5-10 second intervals provides the optimal balance for most applications, offering stable learning while maintaining reasonable responsiveness. Event-driven approaches are most realistic and efficient but require sophisticated implementation. High-frequency decisions should be avoided except in research scenarios specifically studying fine-grained temporal control, as they introduce instability without corresponding performance benefits in traffic control applications.

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
- Action Representation: Phase + Duration (richer control than fixed phases)
- Time Resolution: Flexible parameter for empirical optimization (balance responsiveness vs stability)
- Exploration & Constraints: Enforce minimum green times (realistic, prevents unsafe policies)

## 3. Environment Design

This chapter translates the theoretical problem formulation into a concrete reinforcement learning environment. While Chapter 2 explored the full space of design choices, this chapter focuses on implementing the recommended configuration: centralized agent, macroscopic state representation, global reward design, and phase selection with fixed durations. We cover the practical aspects of environment construction—how to structure the state space, define meaningful rewards and integrate with traffic simulation. The goal is to bridge the gap between conceptual design decisions and working RL implementation, providing the foundation for agent training and performance assessment.

### RL Environment Components

**State space design for macroscopic inputs (per the recommended starting configuration)**: Our RL state space adopts the same macroscopic intermediate values computed by the Tree Method implementation. This design choice enables direct performance comparison between the RL approach and Tree Method on identical traffic state representations, eliminating variability from different input processing pipelines. Since the Tree Method already computes sophisticated traffic flow indicators—average speeds, densities using May's equation, flow rates, and congestion status—we leverage this existing computational framework while adapting the representation for neural network learning.

- **Network Structure Context** (static): Built from NetworkData class (net_data_builder.py:22-45) which extracts edge topology and junction connectivity from .net.xml files. Includes lane configurations per edge with capacity constraints, traffic light phase definitions from junction XML with controlled connections parsed via Link.add_my_phases() (node.py:30-38), free-flow speeds with geometric properties including edge distances and lane counts from Link. (link.py:6-26).

- **Real-Time Traffic State** (updated every decision interval): Average speed per edge using Graph.fill_link_in_step() which calls traci.edge.getLastStepMeanSpeed() (graph.py:124-125), then aggregated via Link.add_speed_to_calculation() (link.py:56-60). Traffic density calculated using May's fundamental diagram in Link.calc_k_by_u(): k = MAX_DENSITY × ((1 - (speed/free_flow_speed))^(1-M))^(1/(L-1)) with parameters MAX_DENSITY, M, L from config.py (link.py:78-79). Flow rate computed as q = speed × density × lanes in Link.calc_my_iteration_data() (link.py:88-90). Congestion status determined by Link.is_loaded = current_speed < q_max_properties.q_max_u where optimal speed comes from Link.calc_max_properties() (link.py:62-73, 91).

- **Signal Control Context** (per intersection): Current active phase retrieved via JunctionNode.save_phase() calling traci.trafficlight.getPhase() (node.py:87-88). Phase transition history tracked in JunctionNode.phases_breakdown_per_iter including switch counts and duration breakdowns (node.py:90-102). Time-loss accumulation computed in Link.calc_my_iteration_data() as actual_time - optimal_time difference, with optimal times from Link.q_max_properties and actual times from current speeds (link.py:100-105).

- **Network Propagation Context** (neighborhood-based): Upstream congestion from Link.links_to_me populated by Link.join_links_to_me() using network connectivity (link.py:28-38). Downstream capacity from Link.links_from_me via Link.join_links_from_me() (link.py:40-49). Bottleneck tree membership tracked in Link.in_tree_fathers and processed by IterationTrees.all_trees_per_iteration, with costs computed via Link.calc_heads_costs() for Tree Method integration (link.py:114-123).

**Action space for dynamic phase control**: The centralized agent simultaneously controls all intersections, selecting discrete phase indices (0 to len(phases)-1) AND continuous duration values for each intersection within safety bounds [MIN_PHASE_TIME, MAX_PHASE_TIME]. Action dimensionality scales as num_intersections × (num_phases + 1) for phase selection plus duration control. This expands the solution space from Tree Method's fixed algorithmic durations to learned optimal timing patterns with direct coordination across the entire network. Implementation through define_tl_program(junction_id, phase_index, learned_duration) with duration clipping for constraint enforcement (node.py:12-14, config.py:2). Safety constraints enforce MIN_PHASE_TIME = 10 seconds minimum green (config.py:2) and maximum phase durations to prevent gridlock, with validation through existing JunctionNode constraint checking (node.py:104-115).

**Reward function**: The reward system implements the theoretical framework from Chapter 2: individual vehicle rewards with global spatial scope, intermediate timing, delta measurement, and time-windowed credit distribution. This translates into a two-component system that provides both dense intermediate learning signals and final episode performance evaluation.

**Two-Component Reward Architecture**:

The reward system combines frequent intermediate penalties based on individual vehicle performance with final episode throughput rewards. At regular measurement intervals (every N seconds), the system evaluates each vehicle's additional waiting time since the last measurement and applies penalties to all network decisions since that vehicle began its journey. At episode completion, the system provides network-wide throughput rewards for vehicles successfully completing their trips.

**Individual Vehicle Tracking and Measurement**:

Each vehicle's journey through the network is tracked from entry to completion, maintaining temporal records of waiting time accumulation. Vehicle tracking leverages SUMO's TraCI interface through traci.vehicle.getIDList() to enumerate active vehicles and traci.vehicle.getAccumulatedWaitingTime() to measure individual waiting times. Delta measurement computes the additional waiting time since the last measurement interval: Δwait_time = current_wait_time - previous_wait_time. This approach captures the immediate impact of recent signal decisions on individual vehicle performance while avoiding double-counting waiting time across measurement periods.

**Intermediate Reward Computation**:

At each measurement interval, the system iterates through all active vehicles, computes their waiting time deltas, and converts these into penalty signals. The reward formula is: penalty = -Δwait_time, where positive waiting time increases result in negative rewards. These penalties are applied to all network decisions made since each vehicle started its journey, implementing the time-windowed credit distribution strategy. Vehicles with longer journey times generate more penalty events, naturally weighting the learning signal toward vehicles most affected by network-wide coordination decisions.

**Credit Distribution Implementation**:

The credit distribution mechanism maintains timestamped records of all signal timing decisions made by the centralized agent. When a vehicle generates a penalty at time T, the system identifies all decisions made since that vehicle's journey start time and applies the penalty to those decisions. This implements global spatial scope (all network locations receive the penalty) with time-windowed temporal scope (only decisions since vehicle start receive credit). The implementation tracks decision_timestamps for each policy action and vehicle_start_times for journey tracking, enabling precise temporal attribution of penalty signals to relevant decision periods.

**Integration with SUMO TraCI and Existing Infrastructure**:

The reward system integrates with existing Tree Method infrastructure through Graph.add_vehicles_to_step() and Graph.close_prev_vehicle_step() methods (graph.py:94-120), extending vehicle tracking to include waiting time history. Vehicle journey completion detection uses Graph.ended_vehicles_count to trigger final throughput rewards. The system maintains compatibility with existing TraCI integration patterns while adding vehicle-specific reward computation capabilities. Decision timestamp tracking integrates with the centralized agent's action execution through define_tl_program() calls (node.py:12-14), creating an audit trail that enables precise credit assignment to specific signal timing decisions.

**Final Episode Throughput Rewards**:

At episode completion, the system provides positive rewards based on network-wide throughput performance: reward = α × completed_vehicles, where α is a scaling factor that balances intermediate penalties with final performance incentives. This component ensures that the agent optimizes for overall system performance rather than simply minimizing individual waiting times. The throughput reward uses existing Graph.ended_vehicles_count tracking to maintain consistency with Tree Method performance measurement approaches.

### Training Environment Implementation

This section covers the practical technical implementation details for building the RL training environment. While the previous section established the conceptual framework—state representation, action spaces, and reward functions—this section focuses on the engineering infrastructure that makes RL training possible. We detail how to bridge the gap between SUMO traffic simulation and reinforcement learning algorithms, covering the essential technical components: simulation integration, data preprocessing, safety enforcement, and training episode management. This is the technical "plumbing" that transforms our conceptual RL design into a working training system.

**SUMO TraCI simulation integration**: The RL environment integrates with the existing dbps system and SUMO simulation through TraCI (Traffic Control Interface). The environment wraps around SUMO's simulation loop, collecting comprehensive network-wide traffic state data through TraCI API calls like traci.edge.getLastStepMeanSpeed() and traci.vehicle.getIDList(), which are already used in Tree Method's Graph.fill_link_in_step() and Graph.add_vehicles_to_step() methods (graph.py:124-137). The centralized agent executes actions across all intersections simultaneously through existing TraCI commands via define_tl_program() function that calls traci.trafficlight.setPhase() and setPhaseDuration() (node.py:12-14). The environment maintains sync with SUMO's simulation steps, collecting complete network observations and applying coordinated RL actions at regular decision intervals while preserving the existing Tree Method infrastructure for fair performance comparison.

**State preprocessing and normalization strategies**: Raw traffic data from Tree Method's calculations must be converted into a single comprehensive neural network input vector for the centralized agent. Traffic speeds from Link.calc_iteration_speed (link.py:56-58) require normalization to [0,1] range using speed limits as maximum bounds. Density values from Link.calc_k_by_u() (link.py:78-79) normalize against MAX_DENSITY parameter. Flow rates from Link.calc_my_iteration_data() (link.py:88-90) need scaling by lane capacity. Vehicle counts from Head.fill_my_count() require normalization by intersection approach capacity. All edge and intersection features are concatenated into a single state vector with dimensionality scaling linearly with network size (num_edges × edge_features + num_intersections × intersection_features). Network-wide features like Graph.ended_vehicles_count and Graph.vehicle_total_time need temporal normalization to handle varying episode lengths.

**Constraint enforcement (minimum green times)**: Safety constraints prevent the RL agent from selecting unsafe or unrealistic signal sequences during training exploration. The environment enforces MIN_PHASE_TIME = 10 seconds (config.py:2) by clipping RL duration actions across all intersections before TraCI execution. Duration values below minimum are automatically adjusted, while maximum phase durations prevent gridlock scenarios. Phase transition validation ensures yellow and all-red clearance intervals are maintained at each intersection simultaneously. The environment integrates with existing JunctionNode constraint checking (node.py:104-115) to validate signal sequences before SUMO execution. Constraint violations trigger action rejection with penalty signals, teaching the agent to respect traffic engineering safety requirements while exploring optimal coordinated timing policies.

**Episode management and environment reset**: Training episodes align with SUMO simulation duration, with each episode representing a complete traffic scenario from initialization to termination. Episode initialization involves SUMO simulation restart through dbps system, network state reset, and vehicle generation restart with consistent random seeds for reproducible training. State reset procedures clear previous episode data from Graph and Link objects while preserving network topology and configuration. Episode statistics collection tracks key training metrics like total completions, average travel times, and reward accumulation for training monitoring and convergence assessment. Environment reset handles simulation warm-up periods to ensure realistic traffic conditions before RL decision-making begins, preventing training on artificial empty-network scenarios.

## 4. Learning Framework Design (Algorithm Selection + Credit Assignment)

Effective reinforcement learning requires two fundamental design decisions that jointly determine how agents learn from experience: the optimization algorithm that updates policy parameters and the credit assignment method that attributes performance outcomes to specific actions. These choices are interdependent - the selected algorithm constrains viable credit assignment strategies, while credit assignment requirements influence algorithm suitability. For network-wide traffic control, both decisions must account for the unique challenges of coordinated multi-intersection control where individual signal timing decisions produce delayed, distributed effects across the entire network.

### RL Algorithm Selection

Algorithm selection for traffic signal control presents unique challenges that distinguish it from standard RL applications. The centralized agent must handle large action spaces and coordinate decisions across multiple intersections simultaneously. The choice of learning algorithm fundamentally determines training feasibility, sample efficiency, and deployment viability for network-wide traffic control.

**Algorithm Categories for Traffic Control**
The choice of optimization algorithm determines how policy parameters are updated based on collected experience. Traffic signal control places specific demands on this choice: mixed discrete-continuous action spaces for phase selection and duration control and expensive simulation episodes that limit sample collectio. These constraints significantly narrow the space of viable algorithms compared to standard reinforcement learning applications.

- **Value-Based Methods (DQN variants)**: Deep Q-Networks learn action-value functions but face significant limitations in traffic control applications. DQN handles discrete action spaces well but struggles with the continuous duration component of our phase+duration action space. Extensions like DDPG for continuous control require separate treatment of discrete phase selection, leading to complex hybrid architectures. Additionally, Q-learning's sample efficiency suffers in high-dimensional action spaces, making training prohibitively expensive when simulation episodes require substantial computational resources.

- **Policy Gradient Methods (PPO, A2C)**: Policy gradient approaches directly optimize policy parameters and naturally handle mixed discrete-continuous action spaces through appropriate output layer designs. PPO's clipped objective provides training stability crucial for large action spaces, preventing destructive policy updates that could destabilize traffic coordination. A2C offers computational efficiency through synchronous updates and lower memory requirements, important considerations for centralized agents processing large state vectors. These methods excel in environments where exploration matters more than precise value estimation.

- **Actor-Critic Approaches**: Hybrid methods combine policy gradient flexibility with value function guidance, offering advantages for traffic control. The actor network naturally outputs both discrete phase selections and continuous durations, while the critic provides variance reduction for policy gradient updates. This architecture aligns well with traffic control intuition: the critic evaluates traffic states (congestion assessment) while the actor decides signal actions (phase timing). Actor-critic methods also support more sophisticated exploration strategies than pure policy gradient approaches.

**Selection Criteria for Traffic Control**
Algorithm selection requires systematic evaluation against criteria that reflect the specific demands of traffic signal control. These criteria prioritize the unique constraints and requirements of network-wide coordination over general machine learning performance metrics.

- **Action Space Compatibility**: The mixed discrete-continuous nature of phase selection and duration control favors policy gradient and actor-critic methods over value-based approaches. Neural network architectures must output discrete probability distributions for phase selection alongside continuous duration values, a requirement naturally handled by policy-based methods.

- **Sample Efficiency**: Traffic simulations are computationally expensive, making sample efficiency critical. Actor-critic methods provide better sample efficiency than pure policy gradients through value function guidance, while policy methods generally require fewer environment interactions than Q-learning for complex action spaces.

- **Training Stability**: Large action spaces and complex coordination requirements demand stable training algorithms. PPO's clipping mechanism prevents large policy updates that could disrupt learned coordination patterns, while A2C's synchronous updates avoid the instability issues of asynchronous methods in centralized control scenarios.

**Recommended Algorithm and Justification**

**Proximal Policy Optimization (PPO)** seems as the optimal choice for centralized traffic control. PPO's actor-critic architecture naturally handles mixed discrete-continuous actions through separate output heads: a softmax layer for phase selection and a continuous output for durations. The clipped objective prevents destructive policy updates that could eliminate beneficial coordination behaviors learned during training. PPO's proven stability and sample efficiency make it well-suited for the expensive simulation environment, while its straightforward implementation facilitates integration with existing traffic simulation infrastructure. The algorithm's robust performance across diverse RL domains provides confidence for traffic control applications where coordination complexity and action space size present significant challenges.

### Credit Assignment Methods

The car-based intermediate reward system from Chapter 3 requires specific algorithmic integration within PPO to ensure stable learning and efficient gradient computation. Unlike traditional RL applications with single reward signals per time step, our system generates multiple simultaneous vehicle penalties at each measurement interval, requiring careful handling of gradient computation and variance control.

**Policy Gradient Integration with Vehicle Penalties**:

PPO's policy gradient computation must accommodate multiple reward signals per time step as different vehicles trigger penalties simultaneously. At each measurement interval, the system collects all vehicle waiting time penalties: R_t = Σ(-Δwait_time_i) for all vehicles i generating penalties at time t. These aggregated penalties integrate into PPO's advantage calculation alongside the final episode throughput reward, creating a two-component gradient signal that balances immediate vehicle performance feedback with long-term network optimization objectives.

**Timestamped Decision Attribution**:

The time-windowed credit distribution strategy requires algorithmic mapping between vehicle penalties and the specific decisions that influenced those vehicles. PPO maintains a decision history buffer storing (timestamp, action, intersection_id) tuples for all signal timing decisions. When a vehicle generates a penalty at time t, the system identifies all decisions made since that vehicle's journey start time and applies the penalty to those specific policy gradient computations. This creates a sparse gradient update pattern where recent decisions receive penalties from multiple vehicles while older decisions receive fewer penalty signals as their affected vehicles complete journeys.

**Variance Reduction for Dense Reward Signals**:

The frequent intermediate penalties can introduce higher gradient variance compared to sparse episode-based rewards. PPO's advantage estimation provides natural variance reduction through the critic network, which learns to predict expected performance from current traffic states. The critic evaluation V(s_t) serves as a baseline for both vehicle penalties and episode rewards: Advantage = R_t - V(s_t), where R_t includes both immediate vehicle penalties and future episode rewards. This baseline subtraction reduces variance from environmental stochasticity while preserving the learning signal from actual policy improvements.

**Training Stability and Reward Balance**:

The two-component reward system requires careful balance between frequent intermediate penalties (typically small, frequent signals) and episode throughput rewards (larger, sparse signals). PPO's clipped objective provides stability against large policy updates from occasional high-magnitude penalty accumulations when multiple vehicles simultaneously experience significant waiting time increases. The reward scaling parameter α for throughput rewards allows empirical tuning to balance immediate vehicle feedback against long-term network performance optimization, ensuring the agent learns both responsive local control and strategic network coordination.

## 5. Implementation Guide

This chapter provides step-by-step implementation instructions for building the RL traffic control system. The implementation extends the existing SUMO/Tree Method infrastructure with a custom OpenAI Gym environment and Stable-Baselines3 PPO training.

### Implementation Overview and File Structure

**Core Implementation Components**:

- `src/rl/traffic_env.py`: Custom Gym environment wrapping SUMO simulation
- `src/rl/vehicle_tracker.py`: Individual vehicle tracking and reward computation
- `src/rl/train.py`: PPO training script using Stable-Baselines3
- `src/rl/evaluate.py`: RL vs Tree Method comparison evaluation

**Integration Points with Existing System**:

- Extends existing `Graph`, `Link`, and `JunctionNode` classes for RL state collection
- Uses existing `define_tl_program()` function for RL action execution
- Leverages existing TraCI integration and SUMO configuration generation

**Required Library Dependencies**:
Install Stable-Baselines3, OpenAI Gym, NumPy, and ensure existing SUMO/TraCI dependencies are available. The implementation builds on the existing infrastructure rather than replacing it.

### Step 1: Custom Gym Environment Implementation

**TrafficControlEnv Class Structure**:
Create a new class inheriting from gym.Env that wraps the existing SUMO simulation. The class requires initialization with the existing configuration system, loading the same Graph and JunctionNode objects used by Tree Method. The environment manages the complete simulation lifecycle including reset, step execution, and termination detection.

**Observation Space Design**:
Define a continuous observation space using gym.spaces.Box with normalized values between 0.0 and 1.0. The observation space dimensions should equal the total state vector size calculated from the number of edges and intersections in the network. Each edge contributes multiple features (speed, density, flow, congestion status) and each intersection contributes phase and timing information.

**Action Space Configuration**:
Implement a continuous action space where each intersection requires two values: phase probability distribution (processed through softmax) and normalized duration value. The total action space dimension equals 2 × number_of_intersections. Actions are processed to select discrete phases and convert normalized durations to actual timing values within safety constraints.

**Core Environment Methods**:
The step() method must execute five sequential operations: process RL actions through TraCI calls, advance the SUMO simulation by one time step, collect new state observations using existing Tree Method calculations, compute rewards through the vehicle tracking system, and determine episode termination based on simulation completion or time limits. The reset() method restarts the SUMO simulation and returns the initial state observation.

### Step 2: Vehicle Tracking and Reward System

**Graph Class Extensions for RL**:
Extend the existing Graph class with RL-specific vehicle tracking capabilities. Add data structures to maintain vehicle journey histories, decision timestamps for credit assignment, and measurement interval tracking. The extensions should integrate seamlessly with existing Graph functionality without disrupting Tree Method operations.

**Vehicle Journey Tracking**:
Implement individual vehicle monitoring that captures vehicle start times, route information, and accumulated waiting times. Each vehicle requires tracking from network entry to completion or episode termination. The tracking system must handle dynamic vehicle populations as vehicles enter and leave the simulation.

**Reward Computation Mechanics**:
Develop the two-component reward system with intermediate vehicle penalties and final episode throughput rewards. Vehicle penalties are computed at regular measurement intervals by calculating waiting time deltas for all active vehicles. The penalty system applies negative rewards proportional to additional waiting time accumulated since the last measurement. Episode rewards provide positive signals based on total vehicle completions.

**Credit Assignment Implementation**:
Build the time-windowed credit distribution system that maps vehicle penalties to relevant signal timing decisions. Maintain timestamp records for all signal control actions and apply penalties to decisions made since each vehicle's journey began. The system ensures that early coordination decisions receive appropriate credit for their long-term effects on vehicle performance.

### Step 3: State Space Implementation

**Integration with Tree Method State Collection**:
Leverage existing Tree Method calculations for traffic state representation. Use the existing Link.calc_my_iteration_data() method that computes speed, density, flow rates, and congestion status. Build the state vector by iterating through all network edges and extracting the computed traffic metrics, ensuring consistency with Tree Method's traffic analysis.

**State Vector Construction**:
Construct the observation vector by concatenating normalized traffic metrics from all edges followed by signal timing information from all intersections. Edge features include normalized speed (current_speed/free_flow_speed), normalized density (density/MAX_DENSITY), normalized flow (flow/capacity), and binary congestion flags. Junction features include normalized current phase and normalized remaining duration.

**Normalization Strategy**:
Implement consistent normalization across all state components to maintain values within the 0.0-1.0 range expected by the observation space. Use network-specific parameters like speed limits, capacity constraints, and timing bounds for normalization. Ensure normalization parameters remain constant across training episodes to maintain learning consistency.

### Step 4: Training Script Setup

**Stable-Baselines3 Integration**:
Create the training script using Stable-Baselines3's PPO implementation with traffic-specific parameter configuration. Initialize the environment using the existing configuration system and validate the implementation using Stable-Baselines3's environment checking utilities before training begins.

**PPO Configuration for Traffic Control**:
Configure PPO with conservative learning parameters appropriate for expensive simulation environments. Use lower learning rates (2e-4) to ensure stable learning with costly episodes. Apply tighter clip ranges (0.1) to prevent large policy updates that could destabilize learned coordination patterns. Set appropriate batch sizes (1024) considering memory constraints from large state vectors.

**Training Pipeline Setup**:
Implement the complete training workflow including environment initialization, model configuration, training execution, and model persistence. Include logging and monitoring capabilities to track training progress, episode rewards, and convergence metrics. Design the pipeline to support training interruption and resumption for long training runs.

### Step 5: Integration Testing and Next Steps

**Incremental Implementation Strategy**:
Follow a systematic approach starting with environment skeleton implementation using dummy rewards to verify basic Gym interface functionality. Progressively add state collection integration, action execution connectivity, vehicle tracking capabilities, reward computation validation, and finally complete PPO training integration.

**Component Testing Approach**:
Develop unit tests for individual components including state collection accuracy, action execution correctness, and reward computation verification. Create integration tests using simple 3×3 grid networks to validate end-to-end functionality before scaling to larger networks. Compare RL environment state collection outputs with Tree Method outputs to ensure consistency.

**Validation and Debugging**:
Implement comprehensive validation checks at each integration step. Verify that state vectors maintain expected dimensionality and normalization ranges. Confirm that action execution produces valid signal timing sequences. Validate that vehicle tracking accurately captures journey information and waiting time calculations. Test reward computation across different traffic scenarios to ensure proper penalty and credit assignment.

## 6. Evaluation

### Evaluation Setup

Compare RL agent vs Tree Method on identical networks and traffic scenarios.

### Metrics

- **Throughput**: Total completed vehicles
- **Waiting time**: Average per vehicle

### Success Criteria

- **Throughput-based**:

- RL achieves >10% improvement in total completed vehicles vs Tree Method
- RL completes >95% of spawned vehicles (high completion rate threshold)

- **Waiting time-based**:

- RL reduces average waiting time by >15% compared to Tree Method
- RL reduces maximum individual vehicle waiting time by >20%
