# Validation

## Validation Framework Overview

- **Purpose**: Comprehensive runtime validation system for ensuring pipeline integrity and correctness
- **Location**: `src/validate/` directory with 4 core modules
- **Usage**: Inline validation at each pipeline step with custom error handling

### errors.py

- **ValidationError**: Custom exception class for validation failures

### validate_network.py

**Network Generation and Processing Validation Functions:**

- **verify_generate_grid_network()**: Validates synthetic grid generation with junction removal
- **verify_insert_split_edges()**: Validates edge splitting into body + head segments
- **verify_split_edges_with_flow_based_lanes()**: Validates comprehensive edge splitting and lane assignment
- **verify_extract_zones_from_junctions()**: Validates zone extraction from junction topology
- **verify_rebuild_network()**: Validates network compilation from separate XML files
- **verify_set_lane_counts()**: Validates lane assignment algorithms and distribution
- **verify_assign_edge_attractiveness()**: Validates attractiveness calculation methods
- **verify_generate_sumo_conf_file()**: Validates SUMO configuration file generation

### validate_traffic.py

**Traffic Generation Validation Functions:**

- **verify_generate_vehicle_routes()**: Validates vehicle route generation with connectivity and statistics checks

### validate_simulation.py

**Simulation Runtime Validation Functions:**

- **verify_tree_method_integration_setup()**: Validates Tree Method traffic control algorithm initialization
- **verify_algorithm_runtime_behavior()**: Validates algorithm behavior during simulation runtime

### validate_split_edges_with_lanes.py

**Edge Splitting and Lane Assignment Validation Functions:**

- **verify_split_edges_with_flow_based_lanes()**: Comprehensive validation of edge splitting and lane assignment

#### Validation Implementation Details:

**Primary Validation Checks:**

1. **Tail Lane Consistency Validation**:
   - **Purpose**: Ensure tail segments retain original edge lane counts
   - **Method**: Compare tail segment lane count against original edge definition
   - **Validation**: `actual_tail_lanes == original_edge_lanes`
   - **Error Example**: "Edge A0B0: tail lanes (1) != original lanes (2)"

2. **Head Lane Accuracy Validation**:
   - **Purpose**: Confirm head segments have correct total movement lanes
   - **Method**: Count total lanes used by all outgoing movements
   - **Validation**: `actual_head_lanes == total_movement_lanes`
   - **Calculation**: `total_movement_lanes = sum(lanes_per_movement for all movements)`
   - **Error Example**: "Edge A0B0: head lanes (3) != total movement lanes (4)"

3. **Movement Integrity Validation**:
   - **Purpose**: Ensure each head lane serves exactly one outgoing movement
   - **Method**: Parse connection file to verify one-to-one movement mapping
   - **Validation**: Each head lane has exactly one outgoing connection
   - **Error Example**: "Edge A0B0: head lane 2 has 3 connections (expected 1)"

4. **Connectivity Validation**:
   - **Purpose**: Verify complete tail-to-head connectivity
   - **Method**: Check all head lanes have incoming connections from tail segments
   - **Validation**: Every head lane receives traffic from at least one tail lane
   - **Error Example**: "Edge A0B0: head lanes {1,3} have no incoming connections from tail"

5. **Complete Flow Validation**:
   - **Purpose**: Ensure all tail lanes connect to head segments
   - **Method**: Verify every tail lane has outgoing connections to head segment
   - **Validation**: No tail lanes are orphaned or disconnected
   - **Error Example**: "Edge A0B0: tail lanes {0,2} have no outgoing connections"

**Structural Validation Checks:**

- **Network Completeness**: Validate all expected tail and head segments exist in network
- **Lane Index Consistency**: Ensure proper sequential lane numbering (0,1,2...)
- **Connection Bounds**: Verify all connections reference valid lane indices within bounds
- **Edge Mapping**: Confirm proper edge-to-movement-to-lane assignments

**Validation Process Flow:**

1. **Network Parsing**: Extract lane information from rebuilt `.net.xml` file
2. **Connection Analysis**: Parse movement data from `.con.xml` file  
3. **Movement Calculation**: Analyze total movement lanes per edge
4. **Cross-Validation**: Compare network structure against connection requirements
5. **Error Reporting**: Generate detailed error messages for any validation failures
6. **Success Confirmation**: Display validation success with edge count

**Integration and Timing:**

- **Execution Point**: After network rebuild (Step 4) when all segments exist
- **CLI Integration**: Seamlessly integrated into pipeline with proper error handling
- **Error Handling**: Uses ValidationError for consistent error reporting
- **Performance**: Validates all edges efficiently in single pass

**Example Validation Output:**

```
Starting comprehensive split edges validation...
✅ VALIDATION PASSED: 80 edges validated successfully
```

**Error Example Output:**

```
❌ VALIDATION FAILED: 3 errors found:
  - Edge A0B0: tail lanes (1) != original lanes (2)
  - Edge C2D2: head lanes (3) != total movement lanes (4)  
  - Edge E1E2: head lane 1 has 0 connections (expected 1)
```

### validate_arguments.py

**CLI Argument Validation Functions:**

- **validate_arguments()**: Validates command-line arguments for consistency and format correctness

**Key Validation Rules:**

- **Grid Dimensions**: Must be between 2-15 (inclusive)
- **Block Size**: Must be 50-500 meters for realistic street networks
- **Vehicle Count**: Must be > 0 and ≤ 10,000 for performance
- **Step Length**: Must be 0.1-10.0 seconds for reasonable simulation granularity
- **Time Parameters**: End time > 0, start time hour 0-24
- **Tree Method Interval**: Must be 30-300 seconds for balanced performance
- **Land Use Block Size**: Must be 10-100 meters following research methodology

**Tree Method Interval Validation:**

The `--tree-method-interval` argument has specialized validation:

- **Range Check**: Validates 30-300 seconds using `CONFIG.TREE_METHOD_MIN_INTERVAL_SEC` and `CONFIG.TREE_METHOD_MAX_INTERVAL_SEC`
- **Performance Guidance**: Lower values (30-60s) provide responsive control with higher CPU usage
- **Efficiency Optimization**: Higher values (120-300s) optimize computation but reduce responsiveness
- **Default Validation**: Uses `CONFIG.TREE_METHOD_ITERATION_INTERVAL_SEC = 90` when argument not provided

**Error Examples:**

```bash
# Invalid interval - too low
ValidationError: Tree Method interval should be 30-300 seconds, got 10

# Invalid interval - too high  
ValidationError: Tree Method interval should be 30-300 seconds, got 500
```

**Integration:**

- **Error Handling**: Uses existing `ValidationError` class for consistent error reporting
- **CLI Integration**: Validation occurs before pipeline execution starts
- **Configuration**: References constants from `src/config.py` for maintainable validation bounds