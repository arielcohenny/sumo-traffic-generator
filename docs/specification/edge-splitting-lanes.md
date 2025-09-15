# Integrated Edge Splitting with Lane Assignment

### 3.1 Unified Edge Splitting Process

#### 3.1.1 Network File Parsing

- **Step**: Parse all existing SUMO XML network files
- **Function**: `split_edges_with_flow_based_lanes()` in `src/network/split_edges_with_lanes.py`
- **Arguments Used**: `--lane_count`, `--seed`

##### File Loading Process:

- **Parse Nodes File**: Load `workspace/grid.nod.xml` for junction coordinates
- **Parse Edges File**: Load `workspace/grid.edg.xml` for edge definitions and geometry
- **Parse Connections File**: Load `workspace/grid.con.xml` for traffic movement analysis
- **Parse Traffic Lights File**: Load `workspace/grid.tll.xml` for signal connections
- **Validation**: Ensure all files exist and are valid XML structures

#### 3.1.2 Movement Analysis

- **Step**: Analyze traffic movements from connection data
- **Function**: `analyze_movements_from_connections()` in `src/network/split_edges_with_lanes.py`
- **Purpose**: Count total outgoing lane movements per edge to determine traffic demand

##### Movement Counting Process:

- **Connection Parsing**: Iterate through all `<connection>` elements in `.con.xml`
- **Lane-Level Counting**: Count total lanes used by all movements for each edge
- **Example**: Edge "A0B0" with connections:
  - To "B0C0" from 1 lane (lane 0)
  - To "B0A1" from 1 lane (lane 0)
  - To "B0B1" from 1 lane (lane 0)
  - **Total movement lanes**: 3 (each movement gets independent capacity after splitting)
- **Pre-Split State**: Before splitting, connections may share lanes (like all movements using lane 0)
- **Post-Split Result**: After edge splitting, each movement gets its own dedicated lane
- **Output**: Dictionary mapping edge IDs to total movement lane counts

#### 3.1.3 Edge Coordinate Extraction

- **Step**: Extract start and end coordinates for each edge
- **Function**: Coordinate extraction logic in `src/network/split_edges_with_lanes.py`

##### Coordinate Extraction Methods:

- **Primary Method**: Extract coordinates from node definitions in `.nod.xml`
- **Node Lookup**: Use edge `from` and `to` attributes to find corresponding nodes
- **Coordinate Assignment**: Assign `(x,y)` coordinates from node positions
- **Fallback Method**: Extract coordinates from edge `shape` attribute if nodes not found
- **Edge Length Calculation**: `edge_length = sqrt((end_x - start_x)² + (end_y - start_y)²)`

#### 3.1.4 Lane Count Determination

- **Step**: Determine lane count for each edge based on network type
- **Function**: Mode-specific lane count determination

##### sample Mode:

- **Lane Count Source**: Extracted from sample data during network import
- **sample Lane Information**: Total number of lanes per edge preserved from original sample data
- **Example**: If sample data specifies edge has 2 lanes, `lane_count = 2`
- **No Algorithm**: Lane count is not calculated, only read from imported network

##### Non-sample Mode (Three Lane Assignment Algorithms):

**1. Realistic Algorithm (`--lane_count realistic`):**

- **Zone Integration**: Load zone data from `workspace/zones.poly.xml`
- **Spatial Analysis**: Find zones adjacent to each edge using geometric intersection
- **Land Use Weights**: Apply traffic generation weights by zone type:
  - Mixed: 3.0 (highest traffic generation)
  - Employment: 2.5 (high peak traffic)
  - Entertainment/Retail: 2.5 (high traffic)
  - Public Buildings: 2.0 (moderate institutional traffic)
  - Residential: 1.5 (moderate residential traffic)
  - Public Open Space: 1.0 (lower recreational traffic)
- **Demand Score Calculation**: Sum weighted zone attractiveness multipliers
- **Perimeter Modifier**: Apply 0.8x modifier for network perimeter edges
- **Lane Mapping**: Score < 1.5 → 1 lane, < 3.0 → 2 lanes, ≥ 3.0 → 3 lanes

**2. Random Algorithm (`--lane_count random`):**

- **Random Generation**: `rng.randint(min_lanes, max_lanes)` using seeded random number generator
- **Bounds**: Respects CONFIG.MIN_LANES (1) and CONFIG.MAX_LANES (3)
- **Reproducibility**: Uses provided seed for consistent results across runs

**3. Fixed Algorithm (`--lane_count [integer]`):**

- **Fixed Assignment**: Uses integer value as lane count for all edges
- **Bounds Checking**: `max(min_lanes, min(max_lanes, fixed_lanes))`
- **Example**: `--lane_count 2` assigns 2 lanes to all edges

#### 3.1.5 Edge Splitting Geometry Calculation

- **Step**: Calculate split points and segment geometry for each edge
- **Function**: Split point calculation in `src/network/split_edges_with_lanes.py`

##### Dynamic Head Distance Calculation:

- **sample and Non-sample Compatible**: `actual_head_distance = min(HEAD_DISTANCE, edge_length/3)`
- **HEAD_DISTANCE Constant**: 50 meters (CONFIG.HEAD_DISTANCE)
- **Short Street Protection**: Prevents geometric issues with short urban streets
- **All Edges Split**: Every edge is always split regardless of length

##### Split Point Geometry:

- **Split Ratio**: `ratio = (edge_length - actual_head_distance) / edge_length`
- **Split Coordinates**:
  - `split_x = start_x + ratio * (end_x - start_x)`
  - `split_y = start_y + ratio * (end_y - start_y)`
- **Segment Creation**:
  - **Tail Segment**: From original start to split point (uses `tail_lanes`)
  - **Head Segment**: From split point to original end (uses `head_lanes`)

##### Unified Lane Assignment Formula (Both sample and Non-sample):

- **Tail Lanes**: `lane_count` (from sample data or calculated algorithm)
- **Head Lanes**: `max(lane_count, total_movement_lanes)`
- **Purpose**: Tail segment uses original lane count, head segment provides adequate capacity for all movements

#### 3.1.6 Movement-to-Head-Lane Assignment Algorithm

- **Step**: Assign each movement to a dedicated head lane (one movement per lane)
- **Function**: Movement assignment logic in `src/network/split_edges_with_lanes.py`
- **Purpose**: Ensure each traffic movement gets its own dedicated lane in the head segment

##### Movement Assignment Process:

**1. Movement Collection:**

- **Extract All Movements**: Parse all outgoing connections from original edge
- **Movement List**: Create ordered list of movements by destination
- **Multi-Lane Movements**: Each movement may use multiple pre-split lanes (e.g., straight movement from lanes 0,1,2)
- **Shared Lane Usage**: Multiple movements may share the same pre-split lanes (e.g., straight, right, and left all using lane 0)
- **Example**: Edge "A0B0" movements: [to_B0C0 from lanes 0,1,2], [to_B0A1 from lane 2], [to_B0B1 from lane 0]

**2. Head Lane Assignment:**

- **Individual Movement Assignment**: Each movement type gets dedicated head lanes equal to the number of pre-split lanes it used
- **Lane-to-Lane Mapping**: Each pre-split lane that served a movement gets its own dedicated head lane for that movement
- **No Lane Sharing**: After splitting, no two movements share the same head lane

**3. Connection Creation:**

- **Head-to-External Connections**: Each head lane connects to its assigned destination
- **Lane Specification**: Connection specifies exact fromLane and toLane indices
- **Example Connections**:
  - `<connection from="A0B0_H" to="B0C0" fromLane="0" toLane="0"/>`
  - `<connection from="A0B0_H" to="B0A1" fromLane="1" toLane="0"/>`
  - `<connection from="A0B0_H" to="B0B1" fromLane="2" toLane="0"/>`

##### Movement Priority Algorithm:

**Movement Ordering Principle:**

- **One Movement Per Head Lane**: Each movement gets exactly one dedicated head lane, regardless of how many lanes it used before splitting
- **Spatial Preference**: Movements are assigned based on turn direction when possible
- **Sequential Assignment**: When spatial preference conflicts occur, use sequential order (0, 1, 2...)

**Example Movement Assignments:**

**Scenario 1: 3 movements from 1 shared lane → 3-lane head:**

- **Movement 1: Straight to B0C0 (originally from lane 0)** → Head lane 1
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 3: Left turn to B0A1 (originally from lane 0)** → Head lane 2

**Scenario 2: 2 movements from 2 different lanes → 2-lane head:**

- **Movement 1: Straight to B0C0 (originally from lane 1)** → Head lane 1
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0

**Scenario 3: 3 movements from 3 lanes → 3-lane head:**

- **Movement 1: Straight to B0C0 (originally from lanes 1,2)** → Head lane 1
- **Movement 3: Right turn to B0B1 (originally from lane 0)** → Head lane 0

**Scenario 4: 4 movements from 3 lanes → 6-lane head:**

- **Movement 1: Straight to B0C0 (originally from lane 0,1,2)** → Head lanes 1,2,3
- **Movement 3: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 4: Left turn to B0A1 (originally from lane 2)** → Head lane 4
- **Movement 5: U-turn to B0A0 (originally from lane 2)** → Head lane 5

**Scenario 5: 3 movements from 3 lanes → 5-lane head:**

- **Movement 1: Straight to B0C0 (originally from lanes 0,1,2)** → Head lane 1,2,3
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 3: Left turn to B0A1 (originally from lane 2)** → Head lane 4

**Scenario 6: 3 movements from 2 lanes → 3-lane head:**

- **Movement 1: Straight to B0C0 (originally from lanes 0)** → Head lane 1
- **Movement 2: Right turn to B0B1 (originally from lane 0)** → Head lane 0
- **Movement 3: Left turn to B0A1 (originally from lane 1)** → Head lane 2

**Key Principle**: The pre-split lane usage is irrelevant - what matters is that we have N distinct movements that each need their own dedicated head lane. The algorithm counts movements, not the lanes they previously used.

#### 3.1.7 Tail-to-Head Lane Connection Algorithm

- **Step**: Create internal connections between tail and head segments
- **Function**: Internal connection generation logic
- **Purpose**: Connect each tail lane to appropriate head lanes for traffic flow continuity

##### Connection Distribution Cases:

**Case 1: Tail Lanes ≤ Head Lanes:**

- **Algorithm**: Each tail lane connects to multiple head lanes (even distribution)
- **Calculation**:
  - `connections_per_tail = head_lanes // tail_lanes`
  - `extra_connections = head_lanes % tail_lanes`
- **Distribution**:
  - Each tail lane gets `connections_per_tail` head lane connections
  - First `extra_connections` tail lanes get one additional head lane connection
- **Example**: 2 tail → 3 head:
  - Tail lane 0 → Head lanes 0, 1 (2 connections)
  - Tail lane 1 → Head lane 2 (1 connection)

**Case 2: Tail Lanes > Head Lanes:**

- **Algorithm**: Multiple tail lanes connect to each head lane
- **Calculation**:
  - `tails_per_head = tail_lanes // head_lanes`
  - `extra_tails = tail_lanes % head_lanes`
- **Distribution**:
  - Each head lane receives connections from `tails_per_head` tail lanes
  - First `extra_tails` head lanes receive one additional tail lane connection
- **Example**: 3 tail → 2 head:
  - Head lane 0 ← Tail lanes 0, 1 (2 tail lanes)
  - Head lane 1 ← Tail lane 2 (1 tail lane)

**Case 3: Tail Lanes = Head Lanes:**

- **Algorithm**: One-to-one mapping
- **Connection**: Tail lane i → Head lane i
- **Example**: 2 tail → 2 head: tail[0]→head[0], tail[1]→head[1]

##### Sequential Lane Assignment:

- **Left-to-Right Order**: Tail lanes connect to head lanes in sequential order (0,1,2...)
- **Continuous Mapping**: Maintains lane continuity for traffic flow
- **Example Detailed Mapping** (2 tail → 3 head):
  - Connection: tail_lane=0, from_lane=0 → head_lane=0, to_lane=0
  - Connection: tail_lane=0, from_lane=0 → head_lane=1, to_lane=0
  - Connection: tail_lane=1, from_lane=1 → head_lane=2, to_lane=0

#### 3.1.8 XML File Updates

- **Step**: Update all four SUMO XML files with split edges and lane assignments
- **Function**: File update functions in `src/network/split_edges_with_lanes.py`

##### 1. Nodes File (.nod.xml) Updates:

- **Add Split Nodes**: Create new intermediate nodes at calculated split points
- **Node Naming Convention**: Original edge "A0B0" creates intermediate node "A0B0_H_node"
- **Node Attributes**: Include coordinates (x,y at split point), junction type, and radius=10.0
- **XML Structure**: `<node id="A0B0_H_node" x="150.0" y="200.0" radius="10.0"/>`
- **Coordinate Calculation**: Split point positioned at HEAD_DISTANCE (50m) from downstream junction

##### 2. Edges File (.edg.xml) Updates:

- **Replace Original Edges**: Remove original single edges completely from XML
- **Add Tail Segments**: Create edges from original start to split point with `tail_lanes`
  - **Tail Naming**: Original edge "A0B0" becomes tail segment "A0B0" (keeps original ID)
  - **Tail Structure**: `<edge id="A0B0" from="A0" to="A0B0_H_node" numLanes="2" speed="13.89" priority="-1"/>`
  - **Tail Attributes**: Preserves original speed, priority; uses calculated tail lane count
- **Add Head Segments**: Create edges from split point to original end with `head_lanes`
  - **Head Naming**: Original edge "A0B0" becomes head segment "A0B0_H"
  - **Head Structure**: `<edge id="A0B0_H" from="A0B0_H_node" to="B0" numLanes="5" speed="13.89" priority="-1"/>`
  - **Head Attributes**: Preserves original speed, priority; uses calculated head lane count (≥ movement count)
- **Shape Coordinates**: Include explicit shape coordinates for geometric accuracy

##### 3. Connections File (.con.xml) Updates:

- **Update External Connections**: Modify existing connections to reference head segments with movement-to-lane assignments
  - **Original**: `<connection from="A0B0" to="B0C0" fromLane="0" toLane="0"/>`
  - **Updated**: `<connection from="A0B0_H" to="B0C0" fromLane="1" toLane="0"/>` (head segment with spatial lane assignment)
- **Add Internal Connections**: Create new connections between tail and head segments using distribution algorithm
  - **Tail-to-Head**: `<connection from="A0B0" to="A0B0_H" fromLane="0" toLane="0"/>`
  - **Distribution**: Each tail lane connects to multiple head lanes based on calculated ratios
- **Movement-Specific Lane Assignment**: Each movement gets dedicated head lanes based on spatial logic
  - **Right turns**: Rightmost head lanes (e.g., lane 0)
  - **Straight movements**: Middle head lanes (e.g., lanes 1,2,3)
  - **Left turns**: Leftmost head lanes (e.g., lane 4)
- **Lane Indexing**: Ensure proper `fromLane` and `toLane` indexing for movement distribution

##### 4. Traffic Lights File (.tll.xml) Updates:

- **Update Connection References**: Modify traffic light connections to reference head segments
  - **Original**: `<connection from="A0B0" to="B0C0" fromLane="0" toLane="0"/>`
  - **Updated**: `<connection from="A0B0_H" to="B0C0" fromLane="1" toLane="0"/>` (matches connections file)
- **Preserve Signal Logic**: Maintain original phase timing and state strings
  - **Phase Structure**: Keep existing `<phase duration="30" state="GGrr"/>` definitions
  - **TLS Programs**: Preserve traffic light program IDs and logic
- **Link Index Adjustment**: Update linkIndex values to match new connection structure
  - **Recalculate Indices**: Adjust linkIndex to correspond to new head segment connections
  - **State String Length**: Ensure state strings match total number of connections after splitting

#### 3.1.9 Edge Splitting Validation

- **Step**: Comprehensive validation of edge splitting and lane assignment
- **Function**: `verify_split_edges_with_flow_based_lanes()` in `src/validate/validate_split_edges_with_lanes.py`
- **Execution**: Runs after network rebuild (Step 4) to validate final network structure

##### Primary Validation Checks:

1. **Tail Lane Consistency**: Verify tail segments have same lane count as original edges
2. **Head Lane Accuracy**: Confirm head segments have correct total movement lanes
3. **Movement Integrity**: Ensure all head lanes have exactly one outgoing direction
4. **Connectivity Validation**: Check all head lanes have incoming connections from tail lanes
5. **Complete Flow**: Verify all tail lanes lead to head lanes

##### Structural Validation Checks:

- **Network Completeness**: Validate all expected tail and head segments exist
- **Lane Index Consistency**: Ensure proper sequential lane numbering (0,1,2...)
- **Connection Bounds**: Verify all connections reference valid lane indices
- **Edge Mapping**: Confirm proper edge-to-movement-to-lane assignments

##### Validation Process:

- **Network Parsing**: Extract lane information from rebuilt `.net.xml` file
- **Connection Analysis**: Parse movement data from `.con.xml` file
- **Cross-Validation**: Compare network structure against connection requirements
- **Error Reporting**: Detailed error messages for any validation failures
- **Success Confirmation**: "✅ VALIDATION PASSED: X edges validated successfully"

##### Validation Timing:

- **Pre-Network Rebuild**: Not executed (head segments don't exist yet)
- **Post-Network Rebuild**: Executed after Step 4 completion
- **Integration**: Seamlessly integrated into CLI pipeline with proper error handling

#### 3.1.10 Edge Splitting Completion

- **Step**: Confirm successful edge splitting with lane assignment
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Successfully completed integrated edge splitting with flow-based lane assignment."

### 3.2 Common Edge Splitting Outputs

- **Files Modified** (both modes):
  - `workspace/grid.nod.xml`: Network nodes with added split points
  - `workspace/grid.edg.xml`: Edges replaced with tail/head segments and lane counts
  - `workspace/grid.con.xml`: Connections updated with internal connections and lane distribution
  - `workspace/grid.tll.xml`: Traffic lights updated to reference head segments
- **Purpose**: Create optimized network structure with appropriate lane capacity for traffic movements
- **Integration**: Unified algorithm works seamlessly with both sample and synthetic grid networks
- **Configuration Constants**:
  - HEAD_DISTANCE: 50 meters
  - MIN_LANES: 1
  - MAX_LANES: 3
  - Junction radius: 10.0 meters
