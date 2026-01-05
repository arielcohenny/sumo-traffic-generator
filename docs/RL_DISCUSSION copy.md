# Reinforcement Learning for Network-Wide Traffic Control

This work explores the application of reinforcement learning to network-wide traffic signal control, with the objective of improving overall throughput and reducing vehicle waiting times. The general problem requires coordinated decision-making across multiple intersections and the ability to generalize to diverse congestion patterns. To make the problem tractable, we begin by examining key formulation choices and adopting a simplified configuration. This configuration translates the theoretical problem into a concrete reinforcement learning environment that can be systematically trained and evaluated. The resulting training process and evaluation framework provide a basis for assessing performance in this simplified setting and for motivating extensions toward more scalable and generalizable approaches.

## 1. Background: The General Problem

Traffic signal control is central to managing urban mobility, where poorly timed signals lead to congestion and longer travel times. Improving overall throughput and reducing waiting times requires coordinated decisions across multiple intersections, since congestion at one junction can propagate downstream.

Reinforcement learning offers a flexible framework for adaptive control under dynamic conditions, but applying it at the network scale introduces additional challenges. Policies must generalize across diverse congestion patterns and intersection configurations rather than memorizing behaviors for individual junctions. Without mechanisms that encourage such generalization, an RL agent risks treating each intersection and traffic pattern as unique, limiting scalability to larger networks.

This combination of domain-specific complexity and machine learning challenges defines the general problem of reinforcement learning for network-wide traffic signal control. Of particular importance are the reward structure choices, which fundamentally determine what coordination behaviors the agent learns and how effectively it can attribute network-wide performance to specific signal timing decisions. A successful RL implementation should demonstrate measurable improvements in throughput and waiting times compared to baseline traffic control methods, while maintaining stable learning across diverse congestion scenarios. To address these challenges in a tractable way, we begin by examining key formulation choices and adopting a simplified configuration for initial experimentation.

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
- Reward Design: Statistical vehicle rewards, global spatial scope, intermediate rewards (step-by-step feedback)
- Action Representation: Phase Selection Only (fixed 10s duration for stable learning)
- Time Resolution: Fixed 10-second intervals
- Exploration & Constraints: Enforce minimum green times (realistic, prevents unsafe policies)

## 3. RL Environment Design

This chapter translates the design choices from Chapter 2 into a concrete RL environment architecture. We focus on the recommended starting configuration: centralized agent with macroscopic state representation, global reward design, and phase control. The goal is to establish the architectural foundation that bridges theoretical formulation with practical implementation, ensuring the RL system can effectively learn network-wide traffic coordination.

### Environment Architecture Overview

The RL environment acts as a wrapper around the existing SUMO simulation infrastructure, providing a standard OpenAI Gym interface while leveraging the sophisticated traffic analysis capabilities already built into the Tree Method system. This architectural choice enables direct performance comparison between RL and Tree Method approaches using identical traffic state representations and simulation conditions.

**System Integration Strategy**: Rather than rebuilding traffic simulation capabilities, the RL environment extends the existing infrastructure. The Tree Method already computes the macroscopic traffic indicators needed for RL state representation—speeds, densities, flow rates, and congestion status. The RL system reuses these calculations while adding the reward computation components necessary for learning. This integration approach ensures consistency between baseline and RL evaluations while minimizing implementation complexity.

### State Space Design

The state representation implements the macroscopic input model from Chapter 2, providing the centralized agent with comprehensive network-wide traffic information in a compact, normalized format suitable for neural network processing.

**Traffic Flow Representation**: Each network edge contributes four normalized traffic indicators that capture current conditions and congestion status. Speed values represent current flow efficiency relative to free-flow conditions. Density measurements using traffic flow theory indicate edge utilization levels. Flow rates capture actual vehicle throughput on each edge. Congestion flags provide binary indicators of bottleneck conditions that require immediate attention.

**Signal State Representation**: Each intersection contributes timing and phase information that enables coordinated signal control. Current active phases indicate the present signal configuration across all network intersections. Phase timing information shows remaining durations and recent transition history. This signal context allows the centralized agent to make informed coordination decisions based on current network-wide signal states.

**Network Connectivity Context**: The state representation includes spatial relationships between network elements to support coordination learning. Upstream and downstream connectivity information enables the agent to understand how local decisions affect neighboring intersections. Traffic propagation patterns from congested areas help identify coordination opportunities for implementing strategies like green wave progression.

**State Vector Construction**: All network information is concatenated into a single normalized vector with fixed dimensionality determined by the network topology. For a network with E edges and J junctions, the state vector has dimensionality E × 6 + J × 2 + 5 (six traffic indicators per edge, two signal features per junction, and five network-level metrics including Tree Method features). This fixed-size representation enables efficient neural network processing while capturing all essential traffic and signal information needed for coordination decisions.

### Action Space Design

The action space implements the phase control choice from Chapter 2, enabling the centralized agent to make coordinated phase selection decisions across all network intersections simultaneously.

**Discrete Phase Selection**: Each intersection requires one discrete control decision: which traffic signal phase to activate. Phase durations are fixed at 10 seconds (aligning with Tree Method's minimum phase time for direct performance comparison). This simplified action space promotes stable learning while maintaining practical traffic control capabilities.

**Centralized Coordination**: The centralized agent simultaneously controls all intersections in the fixed network topology through discrete action selection. This enables direct coordination across the entire network while providing stable learning through the simplified discrete action space.

**Safety Constraint Integration**: Phase durations are constrained by traffic engineering safety requirements with a fixed 10-second minimum green time. This constraint prevents the agent from learning unsafe or unrealistic signal behaviors while maintaining sufficient exploration space for effective phase timing optimization. The constraint enforcement becomes part of the environment interface, ensuring all learned policies remain deployable in real-world systems.

### Reward System

The reward system uses a multi-objective statistical approach that combines aggregate traffic metrics into a balanced reward signal computed at every simulation step. This design emphasizes global network performance while maintaining responsiveness to real-time traffic conditions.

**Multi-Objective Reward Components** (computed every step):

1. **Throughput Reward**: Positive reward per vehicle completion

   - Primary objective: Maximize vehicle completions
   - Provides immediate positive feedback for successful network throughput

2. **Waiting Time Penalty**: Penalty proportional to aggregate waiting time across all vehicles

   - Secondary objective: Minimize aggregate waiting time
   - Coefficient weighted to prevent dominance over other components

3. **Speed Reward**: Reward based on average network speed

   - Tertiary objective: Maintain healthy network flow
   - Normalized by reference speed (typically 50 km/h)

4. **Bottleneck Penalty**: Penalty per bottleneck edge (edges with speed below optimal)

   - Penalizes congested edges requiring immediate attention
   - Coefficient tuned to ensure strong negative correlation with congestion

5. **Excessive Waiting Penalty**: Extra penalty for vehicles experiencing severe delays

   - Targets vehicles waiting beyond threshold (typically 5 minutes)
   - Encourages attention to worst-case vehicle experiences

6. **Insertion Bonus**: Bonus when vehicle insertion queue is below threshold
   - Encourages clearing vehicle insertion queue
   - Promotes network entry efficiency

**Statistical Aggregation**: All reward components use network-wide aggregate statistics rather than individual vehicle tracking. This approach provides stable learning signals while maintaining computational efficiency for large-scale traffic simulation.

**Component Balance**: The specific coefficient values are tuned through empirical validation to ensure no single component dominates the learning signal (see Section 5 for tuned values and validation results).

## 4. Training Framework Strategy

This chapter establishes the strategic approach for training RL agents on network-wide traffic control. We focus on algorithm selection rationale, training configuration principles, and the integration of multi-objective statistical rewards with modern RL algorithms. The framework builds on Chapter 3's environment design to create an effective learning system for coordinated signal control.

### Algorithm Selection Rationale

Selecting the right RL algorithm for traffic control requires careful consideration of domain-specific constraints that distinguish this application from standard RL problems.

**Domain-Specific Algorithm Requirements**: Traffic signal control imposes unique constraints on algorithm choice. The fully discrete action space (phase selection only) enables standard RL algorithms including both value-based and policy gradient methods. Expensive simulation episodes demand high sample efficiency, favoring algorithms with proven performance on discrete action spaces. The centralized coordination requirement favors methods that can handle coordinated multi-intersection control effectively.

**Algorithm Categories and Traffic Control Fit**:

**Value-Based Methods**: DQN can handle discrete phase selection through standard Q-value estimation for each intersection. The discrete action space enables efficient exploration and often provides better sample efficiency than policy gradient methods in traffic control scenarios.

**Policy Gradient Methods**: Direct policy optimization may provide better coordination learning through joint action optimization. These methods can handle the discrete action space naturally while potentially discovering better coordination patterns than value-based approaches.

**Actor-Critic Hybrid Approaches**: Combining policy gradients with value function guidance provides the benefits of direct policy optimization while adding variance reduction through learned value estimates. The actor-critic architecture naturally aligns with traffic control intuition: the critic evaluates traffic states while the actor decides signal timing actions.

**Algorithm Options**: Both DQN variants and PPO become viable choices for centralized traffic control. DQN offers simpler implementation and often better sample efficiency for discrete action spaces, while PPO may provide superior coordination learning through joint action optimization. The choice between them depends on training preferences: DQN for simplicity and sample efficiency, or PPO for potentially better coordination discovery.

### Training Configuration Strategy

Effective training for traffic control requires domain-specific configuration that accounts for expensive simulations, complex coordination requirements, and the unique reward structure established in Chapter 3.

**Training Stability Considerations**: Traffic control training faces several stability challenges that require careful algorithmic configuration. The large action space from centralized control can lead to training instability if policy updates are too aggressive. The multi-objective reward system creates varying gradient magnitudes that need proper balancing through coefficient tuning. The complex coordination requirements mean that small policy changes can have large performance effects, requiring conservative training approaches.

**Sample Efficiency Requirements**: Traffic simulations are computationally expensive, making sample efficiency a critical constraint. Training configurations must maximize learning from limited environment interactions. This requirement favors algorithms with good sample efficiency and training approaches that can leverage existing domain knowledge rather than learning everything from scratch.

**Hyperparameter Strategy for Traffic Control**: Domain-specific hyperparameter choices can significantly improve training effectiveness. Conservative learning rates prevent destabilization of learned coordination patterns. Appropriate batch sizes balance learning stability with memory constraints from large state vectors. Clip ranges should be tuned to prevent large policy updates while maintaining sufficient exploration capacity.

### Multi-Objective Reward Integration

The statistical reward system from Chapter 3 requires specific integration with the chosen RL algorithm to ensure stable learning and balanced multi-objective optimization.

**Aggregate Reward Handling**: The traffic control system combines reward components (throughput, waiting time, speed, bottlenecks, excessive waiting, insertion bonus) into a single scalar signal at each time step. The training framework computes these components using network-wide aggregate statistics, providing stable learning signals without individual vehicle tracking overhead.

**Component Coefficient Tuning**: The multi-objective reward structure requires careful coefficient balancing to ensure no single component dominates the learning signal. Through empirical validation, coefficients are tuned to achieve <60% dominance for any component while maintaining strong correlations with traffic quality metrics (speed correlation >+0.3, bottleneck correlation <-0.3).

**Variance Management**: Step-by-step reward computation provides dense feedback compared to sparse episode-based signals. The statistical aggregation approach reduces gradient variance by averaging across all vehicles rather than tracking individual journeys, promoting stable learning while maintaining responsiveness to traffic conditions.

**PPO Implementation**: The chosen PPO algorithm integrates naturally with the multi-objective reward structure. The value network learns to predict expected cumulative multi-objective rewards, while the policy network learns phase selection strategies that optimize the balanced objective. Conservative learning rates (3e-4 → 5e-6) and clip ranges (0.2) prevent destabilization while allowing effective coordination learning.

## 5. Implementation Overview

This chapter documents the key implementation facts, technology choices, and empirically-tuned parameters used to build the RL traffic control system. Unlike Sections 1-4 which describe design rationale, this section captures the concrete implementation decisions and validated configuration values.

### Technology Stack

**Reinforcement Learning Framework**: The implementation uses Stable-Baselines3 (SB3), a PyTorch-based library providing production-quality RL algorithms. SB3 was chosen for its robust PPO implementation, extensive documentation, and proven performance on continuous control tasks.

**Environment Interface**: The traffic control environment implements the Gymnasium (OpenAI Gym) API, providing standard `reset()`, `step()`, `observation_space`, and `action_space` interfaces. Gymnasium compatibility ensures the environment works with any RL library supporting the Gym standard.

**Deep Learning Backend**: PyTorch serves as the neural network framework, integrated through Stable-Baselines3. The policy and value networks use PyTorch's automatic differentiation for gradient computation and optimization.

**Simulation Integration**: TraCI provides real-time control and state observation of SUMO simulations. The implementation uses TraCI's Python API to execute phase changes, query vehicle states, and collect traffic metrics at each simulation step.

**Supporting Libraries**: NumPy handles numerical operations and array manipulations for state vector construction. Python's built-in logging provides progress tracking and debugging capabilities during training.

### Implementation Architecture

**Pipeline Integration**: The RL environment integrates with the existing SUMO traffic simulation pipeline rather than building a standalone system. When `--traffic_control rl` is specified, the simulation pipeline instantiates the RL environment and connects it to the SUMO simulation configured through standard pipeline steps.

**Environment-Simulation Connection**: The `TrafficControlEnv` class wraps the SUMO simulation, executing the traffic generation pipeline during initialization to create the network and traffic files. The environment maintains a TraCI connection throughout each episode, executing phase changes and collecting observations at every 10-second decision interval.

**Tree Method Reuse**: The implementation leverages the existing Tree Method traffic analysis infrastructure. The `TrafficAnalyzer` class (originally built for Tree Method) computes edge speeds, densities, flows, and bottleneck detection. The RL environment reuses these calculations for state construction rather than reimplementing traffic analysis.

**Dual-Mode Operation**: The system supports both training mode (no pre-trained model) and inference mode (loading existing model). Training mode uses random initial policies and learns through PPO updates. Inference mode loads a saved model and executes the learned policy without exploration.

### Reward Coefficient Values

The multi-objective reward function combines six components with empirically-tuned coefficients validated through correlation analysis:

**Coefficient Values**:

- `REWARD_THROUGHPUT_PER_VEHICLE = 50.0` - Reward per vehicle completion
- `REWARD_WAITING_TIME_PENALTY_WEIGHT = 0.5` - Waiting time penalty multiplier
- `REWARD_SPEED_REWARD_FACTOR = 10.0` - Average speed reward multiplier
- `REWARD_BOTTLENECK_PENALTY_PER_EDGE = 4.0` - Penalty per bottleneck edge
- `REWARD_EXCESSIVE_WAITING_PENALTY = 2.0` - Penalty per vehicle waiting >5min
- `REWARD_INSERTION_BONUS = 1.0` - Bonus when insertion queue <50 vehicles

**Coefficient Rationale**: These values maintain balanced multi-objective optimization where no single component dominates the learning signal. The low waiting penalty weight (0.5) prevents this abundant signal from overwhelming throughput rewards. The high bottleneck penalty (4.0) ensures strong negative correlation with congestion for effective bottleneck identification. The moderate throughput reward (50.0) provides sufficient positive feedback without causing exploitation of edge cases. Validation confirms these coefficients achieve the target balance criteria:

- No single component exceeds 60% of total reward magnitude
- Speed reward maintains strong positive correlation with network speed (>+0.3)
- Bottleneck penalty maintains strong negative correlation with congestion (<-0.3)

**Validated Balance** (1500 vehicles, 3600s simulation):

- Waiting penalty: 58% of total magnitude (within <60% target)
- Bottleneck penalty: 31% of total magnitude
- Speed reward: 14% of total magnitude
- Throughput reward: 9% of total magnitude
- Achieved correlations: +0.684 (speed), -0.708 (bottlenecks)

**Threshold Values**:

- `REWARD_EXCESSIVE_WAITING_THRESHOLD = 300.0` seconds (5 minutes)
- `REWARD_INSERTION_THRESHOLD = 50` vehicles
- `REWARD_SPEED_NORMALIZATION = 50.0` km/h (reference speed)

### Training Configuration

**PPO Hyperparameters**:

- Learning rate: Scheduled decay from 3×10⁻⁴ to 5×10⁻⁶ — Exponential decay prevents overfitting while allowing initial exploration
- Clip range: 0.2 — Standard PPO value balancing policy update magnitude with training stability
- Batch size: 2048 — Large batches reduce variance in long-episode traffic scenarios
- Number of steps: 4096 — Sufficient experience collection before updates for stable gradient estimates
- Number of epochs: 15 — Multiple optimization passes per batch maximize learning from expensive simulation data
- Discount factor (γ): 0.995 — High discount captures long-horizon cascading traffic effects across network
- GAE lambda (λ): 0.98 — High lambda for accurate advantage estimation in long episodes
- Gradient clipping: 0.5 — Prevents destabilization from occasional large gradients in complex traffic states

**Neural Network Architecture**:

- Policy type: MlpPolicy — Standard fully-connected architecture suitable for fixed-size state vectors
- Hidden layers: [256, 256] — Two layers with 256 units each provide sufficient capacity for ~540-dim state space
- Activation function: ReLU — Standard non-linearity preventing vanishing gradients
- Separate policy and value networks — Actor-critic architecture enables independent optimization of each component

**Entropy Coefficient Schedule**:

- Initial: 0.02 — High exploration early in training to discover effective coordination patterns
- Final: 0.001 — Low exploration late in training to exploit learned policy
- Decay: Linear over 500,000 steps — Gradual transition from exploration to exploitation

**Early Stopping**:

- Enabled with 10-evaluation patience — Stops training after 10 evaluations without improvement
- Minimum improvement threshold: 10.0 — Prevents stopping on minor fluctuations in mean reward
- Rationale: Prevents performance degradation from overtraining on expensive simulation episodes

**Training Scale**:

- Default training duration: 100,000 timesteps
- Checkpoint frequency: Every 10,000 timesteps
- State vector size: ~540 dimensions (for 5×5 grid with 80 edges, 25 junctions)
- Action space size: 25-dimensional MultiDiscrete (4 phases per intersection)

**Computational Considerations**:

- Single environment training: ~2.1 minutes per 1,000 timesteps (~476 timesteps/min)
- Model file size: ~5-10 MB (neural network weights)

### Training Environment Configuration

The RL agent trains on a fixed network topology with variable traffic scenarios to learn robust coordination strategies.

**Fixed Network Parameters** (constant across all training):

- Network topology: `--network-seed 42 --grid_dimension 5 --block_size_m 200`
- Lane configuration: `--lane_count realistic`
- Land use: `--land_use_block_size_m 25.0 --attractiveness land_use`
- Traffic signals: `--traffic_light_strategy opposites`
- Simulation: `--step-length 1.0`

**Variable Traffic Parameters** (adjusted periodically during training for scenario diversity):

- Traffic volume: `--num_vehicles` (adjusted for curriculum learning and load testing)
- Episode duration: `--end-time` (adjusted for training efficiency)
- Traffic randomization: `--private-traffic-seed`, `--public-traffic-seed` (changed periodically)
- Driver behavior: `--routing_strategy` (e.g., `'shortest 75 realtime 25'`)
- Fleet composition: `--vehicle_types` (e.g., `'passenger 95 public 5'`)
- Route distributions: `--passenger-routes`, `--public-routes`
- Temporal patterns: `--departure_pattern`, `--start_time_hour`

**Training Rationale**: The fixed network ensures the agent learns to coordinate traffic signals on a consistent 5×5 grid topology. Variable traffic parameters expose the agent to diverse congestion patterns, driver behaviors, and temporal distributions without changing the fundamental coordination problem. This prevents overfitting to specific traffic scenarios while maintaining a well-defined learning task.
