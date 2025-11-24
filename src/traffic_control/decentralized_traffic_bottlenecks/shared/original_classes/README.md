# Original Reference Classes

This directory contains the original Tree Method implementation classes from the reference repository:
https://github.com/nimrodSerokTAU/decentralized-traffic-bottlenecks

## Purpose

These files serve as a reference implementation for comparing with our adapted Tree Method code.
They are preserved exactly as they appear in the original repository to facilitate:

1. **Implementation Verification**: Compare our adapted code with the original to ensure correctness
2. **Bug Investigation**: Reference the original implementation when investigating issues
3. **Feature Understanding**: Study the original design patterns and algorithms

## Files

All 16 Python files from `code/tree_traffic_control/classes/` directory:

- **Core Classes**:
  - `graph.py` - Main graph structure and traffic light updates
  - `node.py` - Junction nodes with `update_traffic_light()` method
  - `link.py` - Road links/edges with traffic flow calculations
  - `phase.py` - Traffic light phase definitions

- **Tree Algorithm**:
  - `iterations_trees.py` - Tree iteration management
  - `current_load_tree.py` - Current load tree structure
  - `load_tree_branch.py` - Tree branch logic

- **Configuration & Data**:
  - `algo_config.py` - Algorithm configuration
  - `run_config.py` - Run configuration
  - `network.py` - Network data structure
  - `net_data_builder.py` - Network data parser

- **Tracking & Statistics**:
  - `head.py` - Head edge tracking
  - `vehicle.py` - Vehicle tracking
  - `statistics.py` - Iteration statistics
  - `print.py` - Data printing/output

- **Utilities**:
  - `tripper.py` - Trip generation

## Important Notes

- **DO NOT MODIFY** these files - they are reference implementations
- These files use the original repository's import structure and may not run directly in our codebase
- Import paths reference `classes.*`, `enums.*`, `config.*`, and `utils.*` from the original repository
- Some files contain executable code at the bottom (e.g., `NetworkData(...)`, `Tripper(...)`)

## Key Differences from Our Implementation

Our adapted code in `src/traffic_control/decentralized_traffic_bottlenecks/` has been modified to:
- Work with our SUMO pipeline and network generation
- Integrate with our configuration system
- Handle our workspace structure
- Support our CLI interface

## Reference

For the complete original codebase and documentation, see:
https://github.com/nimrodSerokTAU/decentralized-traffic-bottlenecks
