# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

[... existing content remains the same ...]

## Memory

- This project is a sophisticated SUMO traffic simulation framework with intelligent grid network generation and dynamic traffic control
- Uses a comprehensive 8-step pipeline for network and traffic generation
- Implements advanced algorithms for zone extraction, land use assignment, and traffic routing
- Designed with multiple architectural patterns including Strategy, Adapter, and Pipeline patterns
- Supports reproducible simulations through seeded random generation
- **Lane Count Algorithms**: Three modes for lane assignment - `realistic` (zone-based demand calculation), `random` (randomized within bounds), and `fixed` (uniform count)
- **Edge Attractiveness Methods**: Five research-based methods (poisson, land_use, gravity, iac, hybrid) with 4-phase temporal system
- **4-Phase Temporal System**: 
  - Research-based bimodal traffic patterns with morning/evening peaks
  - Pre-calculated attractiveness profiles for efficient simulation
  - Real-time phase switching during simulation based on start time
  - Supports both full-day (24h) and rush hour analysis
  - 1:1 time mapping (1 sim second = 1 real second)
- **4-Strategy Routing System**:
  - Four routing strategies: shortest (static), realtime (30s rerouting), fastest (45s rerouting), attractiveness (multi-criteria)
  - Percentage-based vehicle assignment with validation (must sum to 100%)
  - Dynamic rerouting via TraCI for realtime and fastest strategies
  - Integration with existing temporal and attractiveness systems
  - Research-based implementation mimicking GPS navigation apps
- **Advanced Features**:
  - Zone-based traffic demand calculation using land use types and attractiveness values
  - Spatial analysis for edge-zone adjacency detection
  - Research-based land use multipliers and gravity model parameters
  - Modular architecture allowing combination of spatial, temporal, land use, and routing factors
- **Nimrod's Traffic Control Algorithm**: 
  - Uses Nimrod's decentralized traffic control algorithm for dynamic signal optimization
  - Implements intelligent signal control through tree-based method
  - Dynamically adjusts traffic light phases based on real-time traffic bottlenecks
  - Aims to minimize overall traffic congestion by decentralized decision-making
  - Adapts signal timing based on local traffic conditions at each intersection
- **Reminder**: make sure to periodically update CLAUDE.md and README.md to reflect project developments and improvements