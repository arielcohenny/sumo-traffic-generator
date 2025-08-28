# Network Rebuild

## Purpose and Necessity

After edge splitting and lane assignment modifications, the SUMO network must be rebuilt to:

1. **Consolidate XML Changes**: Integrate all modifications from the 4 separate XML files (.nod, .edg, .con, .tll) into a single coherent network file
2. **Generate Final Network**: Create the definitive `grid.net.xml` file that SUMO simulation engine will use
3. **Resolve Dependencies**: Process interdependencies between nodes, edges, connections, and traffic lights that were modified during splitting
4. **Validate Network Integrity**: Ensure all split edges, new intermediate nodes, and updated connections form a valid SUMO network
5. **Enable Simulation**: Produce a complete network file required for vehicle route generation and simulation execution

## Network Rebuild Process

- **Step**: Rebuild SUMO network from all modified XML components
- **Function**: `rebuild_network()` in `src/sumo_integration/sumo_utils.py`
- **Tool Used**: SUMO's `netconvert` utility
- **Input Files**: Modified XML files from edge splitting process:
  - `workspace/grid.nod.xml` (nodes with new intermediate split nodes)
  - `workspace/grid.edg.xml` (edges with tail/head segments)
  - `workspace/grid.con.xml` (connections with movement-specific lane assignments)
  - `workspace/grid.tll.xml` (traffic lights with updated connection references)
- **Output**: `workspace/grid.net.xml` (complete SUMO network ready for simulation)

## Technical Requirements

**Why Rebuilding is Essential:**

- **SUMO Architecture**: SUMO requires a compiled `.net.xml` file that contains the complete network definition
- **Dependency Resolution**: `netconvert` resolves all geometric calculations, lane geometries, and junction logic
- **Coordinate System**: Establishes final projected coordinate system for all network elements
- **Internal Links**: Generates internal lane connections within junctions that connect incoming and outgoing edges
- **Traffic Light Integration**: Properly links traffic light programs with the actual network topology

**Rebuild Command Process:**

- **Netconvert Execution**: Runs SUMO's `netconvert` tool with appropriate parameters
- **File Integration**: Combines all 4 XML input files into single network representation
- **Geometric Processing**: Calculates final lane geometries, junction shapes, and connection curves
- **Validation**: Ensures network is topologically sound and ready for simulation

## Network Rebuild Validation

- **Step**: Verify network rebuild was successful
- **Function**: `verify_rebuild_network()` in `src/validate/validate_network.py`
- **Validation Checks**:
  - **File Existence**: Confirm `workspace/grid.net.xml` was generated successfully
  - **XML Validity**: Ensure output file is valid XML with proper SUMO network structure
  - **Edge Preservation**: Verify all split edges (tail and head segments) exist in final network
  - **Node Integration**: Confirm all intermediate split nodes are properly integrated
  - **Connection Consistency**: Validate all movement-specific lane assignments are preserved
  - **Geometric Integrity**: Check that edge geometries and junction shapes are correctly calculated

## Network Rebuild Completion

- **Step**: Confirm successful network rebuild
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Rebuilt the network successfully."
- **Ready for Next Steps**: Network is now prepared for edge attractiveness assignment