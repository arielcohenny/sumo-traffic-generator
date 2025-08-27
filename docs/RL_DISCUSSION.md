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

### Network-Specific vs. Network-Agnostic Training

**Key Challenge**: Traffic-signal controllers must operate on diverse road networks — from small grids to large irregular topologies. A central question is whether to train models tailored to a specific network or to design a general controller that can be applied across networks. The trade-off is between specialization and performance on one hand, and scalability and transferability on the other.

**Implementation Complexity**:

- Network-specific training is relatively straightforward because the topology is fixed. The state representation and architecture can be customized to that network, which simplifies model design and training. The drawback is zero portability: the trained model works only for that network, and even minor structural changes require retraining.
- Network-agnostic training is more demanding. The model must handle variable-sized inputs and irregular connectivity patterns, often via graph-based or attention-based architectures. This flexibility increases engineering and computational complexity, but it enables a single controller to be applied across many networks.

**Performance Trade-offs**:

- Network-specific models can exploit known geometry and capacity constraints, often achieving higher performance in the target network. They are well-suited for deployment in a fixed, real-world network where generalization is not required.
- Network-agnostic models prioritize portability: the same policy can be applied (or fine-tuned) across different cities or topologies. However, this generalization comes at a cost — they may underperform compared to specialized models, since some capacity to exploit network-specific structure is sacrificed for flexibility.

### Centralized vs. Distributed Agent Architecture

**Coordination vs. Scalability**:

- Centralized agents observe the entire network and can directly optimize coordination across all junctions. However, the state-action space grows exponentially with network size, making training and inference computationally prohibitive for large networks.
- Distributed agents (one per junction, typically with parameter sharing) scale naturally since each agent only sees local state. Coordination must then emerge indirectly, which may be less efficient but keeps computation tractable.

**Credit Assignment**:

- In distributed setups, it is difficult to assign credit when improvements depend on multi-junction cooperation (e.g., a green wave reducing travel times across an arterial). Each local agent receives only partial feedback, which can slow learning.
- Centralized agents sidestep this issue by optimizing a global reward, but must manage delayed and diffuse rewards spread across a vast action space.

**Training Stability**:

- Distributed systems with parameter sharing can suffer instability, as agents update shared weights from diverse local experiences.
- Centralized training avoids this weight conflict, but requires significant compute and careful exploration strategies to prevent the policy from collapsing into local optima.

**Practical Trade-off**:

- Centralized control is feasible for small toy networks or research settings where full coordination is more important than scalability. Exponential scaling with network size.
- Distributed control with parameter sharing is the dominant approach for real-world scenarios, offering scalability with reasonable performance, especially when combined with mechanisms for limited inter-agent communication or graph-based coordination. Linear scaling with network size

### Model Input

The input representation defines how traffic state is observed and determines whether the model learns location-specific behaviors or generalizable coordination patterns. Four common approaches are:

**Microscopic (per-vehicle states)**:

Represents the traffic system at the individual vehicle level, including positions, speeds, and lane assignments for every car in the network.

- Pros: full system detail, precise trajectory prediction.
- Cons: enormous state space, location-specific features, slow training, poor transferability. Real-time inference may require specialized hardware.

**Macroscopic (aggregate edge-level features)**:

Summarizes traffic conditions using aggregate statistics per edge (e.g., queue lengths, mean speeds, densities, flows). This captures the overall state of each road segment without tracking individual vehicles.

- Pros: compact, stable, matches control granularity, highlights network-level patterns, efficient for training and real-time use. Enables real-time control on standard hardware.
- Cons: loses per-vehicle detail and fine-grained intersection dynamics.

**Multi-Scale Hybrid**:

Combines microscopic and macroscopic representations: detailed vehicle-level input is used at critical bottlenecks, while aggregate data represents less congested areas. The model dynamically balances detail and efficiency.

- Pros: adapts representation (microscopic for bottlenecks, macroscopic elsewhere), balances detail and scalability.
- Cons: complex architecture, instability from switching representations.

**Graph-Based Representation**:
Encodes the road network explicitly as a graph, where nodes represent junctions (with signal and flow features) and edges represent roads (with density, flow, and congestion features). This approach typically employs Graph Neural Networks to process the network structure and propagate information between connected intersections. Moderate to high computational requirements.

- Pros: encodes topology directly, enables parameter sharing across junctions, supports transfer to new networks, captures both local and global propagation effects through message passing, and naturally handles irregular network topologies.
- Cons: requires careful feature design, computationally heavier than simpler representations, harder to handle dynamic topology changes, limited by the current maturity of GNN architectures for sequential decision-making, and challenges due to complex message passing dynamics.

### Reward Design

Reward structure fundamentally determines what behaviors the agent learns and directly impacts training efficiency, credit attribution, and performance outcomes. Traffic control presents unique challenges: individual signal decisions have delayed effects, vehicle journeys span multiple intersections, and network performance depends on complex coordination patterns.

**Reward Aggregation: Individual Vehicle vs. Statistical Vehicle Rewards**:

- **Individual vehicle rewards** track each vehicle's journey through the network, providing completion bonuses and journey-specific waiting time penalties. This approach enables natural load balancing across different routes and traffic patterns while providing precise temporal attribution of when performance improvements occur.
- **Statistical vehicle rewards** aggregate performance across all vehicles into network-wide metrics like total throughput, average waiting time, or completion rates. This simplifies implementation but obscures which specific decisions contributed to performance improvements.

**Spatial Scope: Global vs. Local Rewards**:

- **Global rewards** Rewarding the entire network. Encourage system coordination and capture how individual junction decisions influence downstream traffic conditions. However, they suffer from high variance and attribution challenges—all decisions receive identical rewards regardless of individual contribution.
- **Local rewards** Rewarding specific junctions. Provide clearer attribution to specific junctions but create critical blind spots: they fail to capture how one junction's decisions influence other junctions or contribute to overall network performance. This can lead agents to optimize local metrics while harming system-wide coordination.

**Temporal Structure: Episode vs. Intermediate Rewards**:

- **Episode-based rewards** provide final performance evaluation but create sparse learning signals and high gradient variance from delayed feedback.
- **Intermediate rewards** (periodic network measurements during episodes) reduce variance through more frequent feedback but require careful frequency tuning and may encourage short-term optimization.

**Measurement Approach: Cumulative vs. Delta Performance**:

- **Cumulative measurement** tracks total performance from episode start, providing stable long-term signals but potentially double-counting improvements and obscuring recent decision impacts.
- **Delta measurement** rewards performance changes from a chosen reference point (which could be last measurement, or a sliding time window, etc), offering targeted feedback on decision effectiveness but potentially missing decisions taken before the reference point that have later influence—a major limitation in traffic control where signal decisions often have delayed effects spanning multiple signal cycles.

**Credit Distribution Strategy**:

The fundamental challenge is determining which decisions deserve credit when performance improvements occur. Traffic effects are inherently delayed—a signal decision impacts traffic conditions several minutes later—requiring sophisticated temporal credit assignment.

- **All previous decisions**: When rewards are computed, credit goes to all decisions since episode/measurement start, ensuring early coordination decisions receive appropriate recognition.
- **Time-windowed credit**: Limit credit to recent decisions within a fixed time window, balancing recency bias with computational efficiency.

**Research Precedents**: Computer network congestion control uses similar reward structures—TCP algorithms reward window adjustments based on individual connection performance rather than global network metrics. Chess AI (AlphaZero) demonstrates that sparse episode rewards can work with sufficient value function prediction, suggesting hybrid approaches for traffic control.

### Action Representation

**Choice of Action Space**:

- Phase selection: pick which signal phase is active, with fixed durations.
- Phase + duration: select both the active phase and its green time.
- Incremental control: extend or terminate the current phase.

**Trade-off**:

- Richer action spaces offer more flexibility but increase learning difficulty and instability.
- Incremental control tends to be more stable and realistic, since real-world controllers extend or switch rather than reset full cycles.

### Time Resolution / Decision Frequency

**When to Act**:

- Every simulation tick (e.g., 1s) offers maximum control granularity but produces noisy, unstable policies.
- Fixed intervals (e.g., every 5–10s) smooth decisions and align better with signal operations.
- Event-driven (e.g., after minimum green expires) provides adaptive control with fewer unnecessary decisions.

**Trade-off**:

- Higher frequency increases sample complexity and instability.
- Lower frequency improves realism and stability but reduces responsiveness.
- Event-driven: Most efficient, as decisions are made only when necessary, but requires more complex state management.

### Exploration Strategy & Constraints

**Safety Constraints**:

- Traffic signals must obey minimum green times, yellow/all-red clearance, and maximum green durations.
- Policies trained without such constraints risk unsafe or unrealistic behaviors.

**Exploration Choices**:

- Constrained exploration: restricts actions to feasible signal sequences, simplifying training but limiting policy discovery.
- Unconstrained exploration: maximizes flexibility but may waste samples on unsafe or infeasible actions.

**Trade-off**:

- Incorporating constraints during training ensures safety and realism, but care must be taken not to overconstrain and block useful strategies.

### Summary

These formulation choices are deeply interdependent. The training scope shapes the viability of the input model: network-agnostic approaches require graph-based or aggregated features, while network-specific models can rely on simpler encodings. The agent architecture interacts with reward design, since distributed agents need local or hybrid rewards for effective learning, whereas centralized control can optimize a single global objective. Likewise, the action representation and time resolution must align with real-world signal operations, ensuring that decisions are both learnable and realistic. Finally, exploration and constraints influence all other axes, since the boundaries imposed during training determine which strategies the agent can discover. In practice, effective RL controllers emerge not from optimizing each choice in isolation, but from carefully balancing these dimensions to fit the deployment context.

**Recommended Starting Configuration**:

- Training Scope: Network-specific (much faster training, sufficient for proof-of-concept)
- Agent Architecture: Centralized (direct coordination, optimal for proof-of-concept)
- Input Model: Macroscopic (compact state space, stable training)
- Reward Design: Individual vehicle rewards, global spatial scope, intermediate rewards, delta measurement, time-windowed credit distribution
- Action Space: Phase + Duration (richer control than fixed phases)
- Decision Frequency: Flexible parameter for empirical optimization (balance responsiveness vs stability)
- Constraints: Enforce minimum green times (realistic, prevents unsafe policies)

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

## 5. Training Implementation

- Large action space exploration strategies
- Computational requirements for centralized control
- Hyperparameters and curriculum learning

## 5. Evaluation

Evaluation should be to the tree method
