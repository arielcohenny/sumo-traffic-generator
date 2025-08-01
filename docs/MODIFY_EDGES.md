# Custom Edge Lane Definition System - Complete Implementation Plan

## Overview
Add capability for users to manually specify lane configurations for specific edges in synthetic grids, overriding the automatic lane assignment algorithms. This is a post-processing step that occurs after the current `split_edges_with_flow_based_lanes()` completes.

## Flexible Configuration Syntax

### Full Specification (Most Common)
```bash
# Complete edge customization: explicit tail lanes + head movements
--custom_lanes "A1B1=tail:2,head:B1B0:1,B1C1:2;A2B2=tail:3,head:B2C2:2,B2B1:1"
```

### Tail-Only Customization
```bash
# Only change tail lanes, preserve all existing head movements
--custom_lanes "A1B1=tail:2;A2B2=tail:3"
```

### Head Movements Only  
```bash
# Only customize head movements, keep automatic tail assignment
--custom_lanes "A1B1=head:B1B0:1,B1C1:2;A2B2=head:B2C2:2,B2B1:1"
```

### Dead-End Creation
```bash
# Create dead-end streets (vehicles can enter but not exit)
--custom_lanes "A1B1=tail:2,head:;A2B2=tail:3,head:"
```

### Mixed Approaches
```bash
# Different customization levels for different edges
--custom_lanes "A1B1=tail:2;A2B2=head:B2C2:2,B2B1:1;B1C1=tail:3,head:C1D1:2"
```

### Alternative File-Based Approach
```bash
# For complex scenarios with many customizations
--custom_lanes_file custom_lanes.txt
```

## File-Based Configuration Specification

### `--custom_lanes_file` Complete Documentation

#### File Format
The file uses the same syntax as the CLI argument, with one edge configuration per line:

```
# custom_lanes.txt - Example configuration file
# Comments start with # and are ignored
# Empty lines are ignored

# Full specification examples
A1B1=tail:2,head:B1B0:1,B1C1:2
A2B2=tail:3,head:B2C2:2,B2B1:1

# Tail-only customizations
B1C1=tail:2
B2C2=tail:3

# Head movements only
C1D1=head:D1D2:1,D1E1:2
C2D2=head:D2D3:2,D2E2:1

# Dead-end creation
D1E1=tail:2,head:
D2E2=tail:3,head:

# Mixed approaches on same line allowed
E1F1=tail:1;E2F2=head:F2G2:2;E3F3=tail:2,head:F3G3:1,F3F4:1
```

#### File Location and Path Handling
- **Relative paths**: Resolved relative to current working directory
- **Absolute paths**: Used as-is
- **File must exist**: Non-existent files cause immediate error and exit
- **File must be readable**: Permission errors cause immediate error and exit

#### Example Usage
```bash
# Using relative path
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --custom_lanes_file configs/intersection_test.txt \
  --gui

# Using absolute path  
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --custom_lanes_file /home/user/experiments/custom_lanes_config.txt \
  --gui

# Cannot use both CLI and file options simultaneously
# This will cause validation error:
env PYTHONUNBUFFERED=1 python -m src.cli \
  --custom_lanes "A1B1=tail:2" \
  --custom_lanes_file config.txt \  # ERROR: mutually exclusive
  --gui
```

#### File Content Validation
- **Line-by-line parsing**: Each line processed independently
- **Comment handling**: Lines starting with `#` are ignored
- **Empty line handling**: Blank lines are ignored
- **Error reporting**: Line numbers included in error messages
- **Same validation rules**: All CLI syntax validation applies to file content
- **Encoding**: UTF-8 encoding required

#### Error Handling
```bash
# File not found
ERROR: Custom lanes file does not exist: configs/missing.txt

# Permission denied
ERROR: Cannot read custom lanes file: /restricted/config.txt

# Invalid syntax on line 15
ERROR: Invalid custom lanes syntax on line 15: "A1B1=invalid_format"

# Invalid edge ID on line 8
ERROR: Invalid edge ID format on line 8: "INVALID123" - Must match pattern A1B1, B2C2, etc.
```

## Understanding Current Edge Structure

After edge splitting, edges are transformed as follows:
- **Original edge**: `A1B1` becomes:
  - **Tail segment**: `A1B1` (from A1 to A1B1_H_node)
  - **Head segment**: `A1B1_H_s` (from A1B1_H_node to B1)
  - **Intermediate node**: `A1B1_H_node` (split point)

## Complete XML File Changes Required

### 1. `edg.xml` Changes
**Target**: Modify `numLanes` attribute of existing split edges

**Example**: User specifies `A1B1=tail:2,head:B1B0:1,B1C1:2`
```xml
<!-- BEFORE custom lanes -->
<edge id="A1B1" from="A1" to="A1B1_H_node" numLanes="1" speed="13.89" priority="-1" 
      shape="100.0,200.0 100.0,250.0" />
<edge id="A1B1_H_s" from="A1B1_H_node" to="B1" numLanes="4" speed="13.89" priority="-1" 
      shape="100.0,250.0 100.0,300.0" />

<!-- AFTER custom lanes -->  
<edge id="A1B1" from="A1" to="A1B1_H_node" numLanes="2" speed="13.89" priority="-1" 
      shape="100.0,200.0 100.0,250.0" />                                  <!-- Updated tail: numLanes only -->
<edge id="A1B1_H_s" from="A1B1_H_node" to="B1" numLanes="3" speed="13.89" priority="-1" 
      shape="100.0,250.0 100.0,300.0" />                                  <!-- Updated head: numLanes 4→3 (ONLY specified movements) -->
```

**CRITICAL**: Only `numLanes` attribute is modified. All other attributes (`speed`, `priority`, `shape`) remain unchanged to preserve existing network geometry and traffic characteristics.

### 2. `con.xml` Changes
**CRITICAL STRATEGY**: Complete Deletion and Regeneration

**DELETE ALL CONNECTIONS** for the customized edge:
```xml
<!-- Remove all existing connections FROM the tail segment -->
<connection from="A1B1" to="A1B1_H_s" fromLane="..." toLane="..." />

<!-- Remove all existing connections FROM the head segment -->  
<connection from="A1B1_H_s" to="B1B0" fromLane="..." toLane="..." />
<connection from="A1B1_H_s" to="B1C1" fromLane="..." toLane="..." />
<connection from="A1B1_H_s" to="B1B2" fromLane="..." toLane="..." />
<connection from="A1B1_H_s" to="B1A1" fromLane="..." toLane="..." />

<!-- CRITICAL: Also remove connections TO the tail segment (bidirectional impact) -->
<connection from="A0A1_H_s" to="A1B1" fromLane="0" toLane="0" />
<connection from="A0A1_H_s" to="A1B1" fromLane="1" toLane="1" />
<connection from="A0A1_H_s" to="A1B1" fromLane="2" toLane="2" />  <!-- May be invalid after tail reduction -->
```

**REGENERATE COMPLETELY NEW CONNECTIONS**:
```xml
<!-- New internal tail→head connections (2 tail lanes → 3 head lanes) -->
<connection from="A1B1" to="A1B1_H_s" fromLane="0" toLane="0"/>
<connection from="A1B1" to="A1B1_H_s" fromLane="1" toLane="1"/>
<connection from="A1B1" to="A1B1_H_s" fromLane="1" toLane="2"/>

<!-- New head→downstream connections - ONLY USER-SPECIFIED MOVEMENTS -->
<!-- CRITICAL: Original movements B1B2 and B1A1 are COMPLETELY DELETED -->
<!-- User specified in head: section: B1B0 (1 lane) + B1C1 (2 lanes) -->

<connection from="A1B1_H_s" to="B1B0" fromLane="0" toLane="0"/>         <!-- RIGHT: 1 lane (USER SPECIFIED) -->
<connection from="A1B1_H_s" to="B1C1" fromLane="1" toLane="0"/>         <!-- STRAIGHT: 2 lanes (USER SPECIFIED) -->
<connection from="A1B1_H_s" to="B1C1" fromLane="2" toLane="1"/>         <!-- STRAIGHT: 2nd lane -->

<!-- B1B2 and B1A1 movements NO LONGER EXIST - these destinations are unreachable from A1B1! -->

<!-- Head segment now has 3 lanes total: 1(right) + 2(straight) = 3 lanes -->

<!-- New upstream→tail connections (redistributed for new tail lane count) -->
<connection from="A0A1_H_s" to="A1B1" fromLane="0" toLane="0"/>         <!-- Redistributed -->
<connection from="A0A1_H_s" to="A1B1" fromLane="1" toLane="1"/>         <!-- Redistributed -->
<connection from="A0A1_H_s" to="A1B1" fromLane="2" toLane="1"/>         <!-- Lane 2 traffic merged to lane 1 -->
```

### 3. `nod.xml` Changes
**Strategy**: No changes required (nodes already exist from edge splitting)

### 4. `tll.xml` Changes - THE MOST COMPLEX PART
**Strategy**: Delete and recreate affected traffic light logic AND connections

#### 4a. Traffic Light Logic State Recalculation (Top of file)
**CRITICAL**: When connection count changes at a junction, the state strings must be recalculated!

```xml
<!-- BEFORE: B1 has 16 connections, state="GGggrrrrGGggrrrr" (16 chars) -->
<tlLogic id="B1" type="static" programID="0" offset="0">
    <phase duration="42" state="GGggrrrrGGggrrrr" />
    <phase duration="3" state="yyyyrrrryyyyrrrr" />  
    <phase duration="42" state="rrrrGGggrrrrGGgg" />
    <phase duration="3" state="rrrryyyyrrrryyyy" />
</tlLogic>

<!-- AFTER: B1 now has 14 connections, need state="GgrrrrGGgrrrr" (14 chars) -->
<tlLogic id="B1" type="static" programID="0" offset="0">
    <phase duration="42" state="GgrrrrGGgrrrr" />     <!-- RECALCULATED! (2 fewer connections) -->
    <phase duration="3" state="yyrrrryyyyrrrr" />      <!-- RECALCULATED! (2 fewer connections) -->
    <phase duration="42" state="rrGGggrrrrGGgg" />     <!-- RECALCULATED! (2 fewer connections) -->
    <phase duration="3" state="rryyyrrrryyy" />       <!-- RECALCULATED! (2 fewer connections) -->
</tlLogic>
```

#### 4b. Traffic Light Connection linkIndex Recalculation (Bottom of file)
**CRITICAL**: ALL connections going into affected junction need linkIndex recalculated!

**BEFORE Custom Lanes (Junction B1 - 16 total connections)**:
```xml
<!-- Original traffic light connections (4 from A1B1_H_s) -->
<connection from="A1B1_H_s" to="B1B0" fromLane="0" toLane="0" tl="B1" linkIndex="0" />
<connection from="A1B1_H_s" to="B1C1" fromLane="1" toLane="0" tl="B1" linkIndex="1" />
<connection from="A1B1_H_s" to="B1B2" fromLane="2" toLane="0" tl="B1" linkIndex="2" />
<connection from="A1B1_H_s" to="B1A1" fromLane="3" toLane="0" tl="B1" linkIndex="3" />
<!-- All other connections to B1 -->
<connection from="B0B1_H_s" to="B1C1" fromLane="0" toLane="0" tl="B1" linkIndex="4" />
<connection from="B0B1_H_s" to="B1B2" fromLane="1" toLane="0" tl="B1" linkIndex="5" />
<connection from="B0B1_H_s" to="B1A1" fromLane="2" toLane="0" tl="B1" linkIndex="6" />
<connection from="B0B1_H_s" to="B1B0" fromLane="3" toLane="0" tl="B1" linkIndex="7" />
<!-- ... linkIndex 8-15 for other incoming edges ... -->
```

**AFTER Custom Lanes (Junction B1 - 14 total connections)**:
```xml
<!-- CRITICAL: con.xml and tll.xml connections MUST BE IDENTICAL -->
<!-- Only difference: tll.xml adds tl="B1" and linkIndex attributes -->

<!-- New traffic light connections (3 from A1B1_H_s) - MATCHES con.xml exactly -->
<connection from="A1B1_H_s" to="B1B0" fromLane="0" toLane="0" tl="B1" linkIndex="0" />    <!-- RIGHT: 1 lane -->
<connection from="A1B1_H_s" to="B1C1" fromLane="1" toLane="0" tl="B1" linkIndex="1" />    <!-- STRAIGHT: 2 lanes -->
<connection from="A1B1_H_s" to="B1C1" fromLane="2" toLane="1" tl="B1" linkIndex="2" />    <!-- STRAIGHT: 2nd lane -->

<!-- B1B2 and B1A1 connections DELETED - no longer exist! -->

<!-- ALL OTHER CONNECTIONS SHIFTED -2 in linkIndex (2 fewer connections) -->
<connection from="B0B1_H_s" to="B1C1" fromLane="0" toLane="0" tl="B1" linkIndex="3" />    <!-- Was 4, now 3 -->
<connection from="B0B1_H_s" to="B1B2" fromLane="1" toLane="0" tl="B1" linkIndex="4" />    <!-- Was 5, now 4 -->
<connection from="B0B1_H_s" to="B1A1" fromLane="2" toLane="0" tl="B1" linkIndex="5" />    <!-- Was 6, now 5 -->
<connection from="B0B1_H_s" to="B1B0" fromLane="3" toLane="0" tl="B1" linkIndex="6" />    <!-- Was 7, now 6 -->
<!-- ... ALL remaining connections linkIndex -2 ... -->
```

**Key Requirement**: linkIndex assignment is SEQUENTIAL (0,1,2,3...) for ALL connections entering junction B1

## Bidirectional Impact Analysis

**CRITICAL INSIGHT**: Changing lanes on ONE edge has BIDIRECTIONAL impacts affecting multiple junctions!

### When we modify `A1B1=tail:2,head:B1B0:1,B1C1:2`, we must update:

#### Direct Impact (Outgoing - Junction B1):
1. **All connections FROM A1B1_H_s TO other edges** (modified movements)
2. **Traffic light state string for B1** (connection count may change)  
3. **All linkIndex values for B1** (sequential reassignment)

#### Bidirectional Impact (Incoming - Junction A1):  
4. **All connections FROM other edges TO A1B1** (tail lane reduction)
5. **Traffic light state string for A1** (if connection count changes)
6. **All linkIndex values for A1** (if connections redistributed)

### Complete Junction Recalculation Required:

**Junction B1** (downstream):
- `A1B1_H_s` → B1 (modified outgoing movements)
- `B0B1_H_s` → B1 (linkIndex shifted)  
- `B2B1_H_s` → B1 (linkIndex shifted)
- `C1B1_H_s` → B1 (linkIndex shifted)

**Junction A1** (upstream):  
- `A0A1_H_s` → A1 → A1B1 (incoming lane redistribution)
- `A2A1_H_s` → A1 → A1B1 (incoming lane redistribution)
- `B1A1_H_s` → A1 → A1B1 (incoming lane redistribution)

## Complete Traffic Light System Impact

### The Cascade Effect: How One Edge Change Affects Entire Traffic System

**Example**: Customizing `A1B1=tail:2,head:B1B0:1,B1C1:2` creates this cascade:

#### 1. Connection Count Changes
```xml
<!-- BEFORE: A1B1_H_s has 4 outgoing connections (B1B0, B1C1, B1B2, B1A1) -->
<!-- AFTER: A1B1_H_s has 3 outgoing connections (ONLY B1B0 + B1C1 specified by user) -->
<!-- B1B2 and B1A1 movements are COMPLETELY DELETED -->
```

#### 2. Junction B1 Traffic Light Logic Recalculation
```xml
<!-- BEFORE: Junction B1 - 16 total connections -->
<tlLogic id="B1" type="static" programID="0" offset="0">
    <phase duration="42" state="GGggrrrrGGggrrrr" />  <!-- 16 characters -->
    <phase duration="3" state="yyyyrrrryyyyrrrr" />   <!-- 16 characters -->
    <phase duration="42" state="rrrrGGggrrrrGGgg" />  <!-- 16 characters -->
    <phase duration="3" state="rrrryyyyrrrryyyy" />   <!-- 16 characters -->
</tlLogic>

<!-- AFTER: Junction B1 - 14 total connections (2 fewer) -->
<tlLogic id="B1" type="static" programID="0" offset="0">
    <phase duration="42" state="GgrrrrGGgrrrr" />  <!-- 14 characters - 2 fewer! -->
    <phase duration="3" state="yyrrrryyyyrrrr" />   <!-- 14 characters - 2 fewer! -->
    <phase duration="42" state="rrGGggrrrrGGgg" />  <!-- 14 characters - 2 fewer! -->
    <phase duration="3" state="rryyyrrrryyy" />    <!-- 14 characters - 2 fewer! -->
</tlLogic>
```

#### 3. All Connection linkIndex Values Shift
```xml
<!-- Every connection after the modified edge gets linkIndex -2 (2 fewer connections) -->
<!-- This affects EVERY edge going into Junction B1, not just A1B1! -->

<!-- B0B1_H_s connections: linkIndex 4→3, 5→4, 6→5, 7→6 -->
<!-- B2B1_H_s connections: linkIndex 8→6, 9→7, 10→8, 11→9 -->
<!-- C1B1_H_s connections: linkIndex 12→10, 13→11, 14→12, 15→13 -->
```

#### 4. State String Character Meaning Changes
```xml
<!-- BEFORE: Position 3 in state string = A1B1_H_s to B1A1 connection -->
<!-- AFTER: Position 3 in state string = B0B1_H_s first connection -->
<!-- Positions 3 and 4 (B1B2 and B1A1) are COMPLETELY REMOVED from state string -->

<!-- EVERY position in the state string now refers to a different connection! -->
```

#### 5. Implementation Requirement: Complete Junction Rebuild
```python
def update_traffic_lights_for_junction(junction_id):
    # 1. Delete ALL traffic light connections for this junction
    # 2. Delete traffic light logic for this junction
    # 3. Count ALL connections going into this junction (from ALL edges)
    # 4. Assign new sequential linkIndex (0,1,2,3...)
    # 5. Generate new state strings based on total connection count
    # 6. Recreate traffic light logic with new state strings
    # 7. Recreate ALL traffic light connections with new linkIndex
```

**CRITICAL**: This is why the complexity is VERY HIGH - one edge change requires complete traffic light system regeneration for affected junctions!

## Dead-End Street Creation

### What `head:` Means
When you specify `head:` with no movements, you create a **dead-end street**:

```bash
# Create dead-end: vehicles can enter A1B1 but cannot exit
--custom_lanes "A1B1=tail:2,head:"
```

### Dead-End Behavior
- **Tail segment**: Functions normally (vehicles can enter from upstream)
- **Head segment**: Has 0 lanes and no outgoing connections
- **Traffic impact**: Vehicles entering A1B1 have nowhere to go (trapped)
- **Use case**: Forces traffic to use alternative routes

### Dead-End XML Changes
```xml
<!-- edg.xml: Head segment gets 0 lanes -->
<edge id="A1B1_H_s" from="A1B1_H_node" to="B1" numLanes="0" ... />

<!-- con.xml: No outgoing connections from head segment -->
<!-- Only internal tail→head connections exist -->
<connection from="A1B1" to="A1B1_H_s" fromLane="0" toLane="0"/>
<connection from="A1B1" to="A1B1_H_s" fromLane="1" toLane="0"/>

<!-- tll.xml: Junction B1 has fewer connections (all A1B1_H_s connections removed) -->
```

### Use Cases for Dead-Ends
1. **Construction Zones**: Temporarily block streets during construction
2. **Cul-de-sacs**: Create residential dead-end streets
3. **Parking Areas**: Streets that only serve parking (no through traffic)
4. **Emergency Testing**: Block routes to test emergency evacuation plans
5. **Traffic Control**: Force congestion to test Tree Method under stress
6. **Network Isolation**: Isolate parts of network for algorithm testing

## Shared Code Architecture - No Duplication Strategy

### Reuse Existing Functions from `split_edges_with_lanes.py`:
```python
# src/network/custom_lanes.py (new file)
from src.network.split_edges_with_lanes import (
    analyze_movements_from_connections,    # Determine available movements
    calculate_movement_angles,             # Calculate actual turn angles  
    assign_lanes_by_angle                  # Assign lanes based on spatial logic
)

def apply_custom_lane_configs(custom_configs):
    """Thin wrapper around existing spatial logic engine."""
    for edge_id, config in custom_configs.items():
        # 1. REUSE existing movement analysis
        movement_data = analyze_movements_from_connections(con_root)
        movements = movement_data[edge_id]['movements']
        
        # 2. REUSE existing angle calculation  
        movements_with_angles = calculate_movement_angles(
            edge_coords[edge_id], movements, edge_coords
        )
        
        # 3. Filter movements to ONLY user-specified ones
        custom_movements = []
        for movement in movements_with_angles:
            to_edge = movement['to_edge']
            if to_edge in config:
                movement['num_lanes'] = config[to_edge]  # CUSTOM OVERRIDE
                custom_movements.append(movement)
        # All other movements are DELETED - not included in custom_movements
        
        # 4. REUSE existing spatial assignment logic on FILTERED movements
        head_lanes = sum(m['num_lanes'] for m in custom_movements)
        movement_to_head_lanes = assign_lanes_by_angle(custom_movements, head_lanes)
        
        # 5. Complete deletion and regeneration
        delete_all_connections_for_edge(edge_id)
        regenerate_connections_with_custom_assignments(edge_id, movement_to_head_lanes)
        update_bidirectional_impacts(edge_id, config)
```

### Key Benefits:
- **No Code Duplication**: Complex spatial logic remains in ONE place
- **Consistent Behavior**: Same spatial assignments for automatic and custom
- **Maintainable**: Single source of truth for spatial logic
- **Thin Wrapper**: Custom lanes = configuration layer over existing engine

## Implementation Architecture

### File Structure
```
src/network/custom_lanes.py          # Main implementation (NEW FILE)
src/config.py                        # Add CustomLaneConfig class
src/args/parser.py                   # Add CLI arguments
src/validate/validate_arguments.py   # Add validation
src/pipeline/standard_pipeline.py    # Integration point
```

### Customization Type Handling

#### 1. Full Specification: `A1B1=tail:2,head:B1B0:1,B1C1:2`
- Custom tail lanes + explicit head movement assignments
- Complete edge reconfiguration
- Bidirectional impact (both junctions affected)

#### 2. Tail-Only: `A1B1=tail:2`  
- Custom tail lanes, preserve all existing head movements
- Minimal impact scope
- Only upstream junction affected (incoming connections)

#### 3. Head Movements Only: `A1B1=head:B1B0:1,B1C1:2`
- Automatic tail lanes, custom movement assignments
- Only downstream junction affected (outgoing connections)
- Tail-to-head connections remain automatic

#### 4. Dead-End Creation: `A1B1=tail:2,head:`
- Custom tail lanes, remove ALL head movements
- Creates dead-end street (no exit possible)
- Significant traffic impact (vehicles get trapped)

### Detailed Algorithm

#### Phase 1: Parse and Validate Custom Specifications
1. Parse flexible syntax (tail-only, head-only, full)
2. Validate edge existence in current XML files
3. Validate movement specifications match available movements
4. Validate lane counts are within bounds (1-3)
5. Validate reachability and flow conservation

#### Phase 2: Calculate Bidirectional Impact Scope
1. For each custom edge, identify affected junctions (upstream AND downstream)
2. Calculate new head lane requirements based on movement specifications
3. Determine lane redistribution strategies for tail changes
4. Assess junction-wide recalculation requirements

#### Phase 3: Update XML Files with Complete Deletion/Regeneration
1. **Update `edg.xml`**: Modify `numLanes` for tail and/or head segments
2. **Update `con.xml`**: 
   - Delete ALL connections for affected edges (bidirectional)
   - Regenerate internal tail→head connections with new distributions
   - Regenerate head→downstream connections using shared spatial logic + custom overrides
   - Regenerate upstream→tail connections with lane redistribution
3. **Update `tll.xml`**:
   - For each affected junction (upstream AND downstream):
     - Count total connections going into junction
     - Regenerate state strings for all phases
     - Delete all traffic light connections for junction
     - Recreate all traffic light connections with sequential linkIndex

#### Phase 4: Spatial Logic Preservation via Shared Functions
1. **Calculate turn angles** using existing `calculate_movement_angles()`:
   ```python
   movements_with_angles = calculate_movement_angles(
       edge_coords[edge_id], movements, edge_coords
   )
   # Result: [{'to_edge': 'B1B0', 'turn_angle': -90.0}, ...]
   ```

2. **Filter to ONLY user-specified movements**:
   ```python
   custom_movements = []
   for movement in movements_with_angles:
       if movement['to_edge'] in custom_config:
           movement['num_lanes'] = custom_config[movement['to_edge']]  # Custom override
           custom_movements.append(movement)
   # All other movements are DELETED - not included in result
   ```

3. **Assign lanes spatially** using existing `assign_lanes_by_angle()`:
   ```python
   # Sorts by angle and assigns lanes: right→lane 0, straight→middle, left→high lanes
   head_lanes = sum(m['num_lanes'] for m in custom_movements)
   movement_to_head_lanes = assign_lanes_by_angle(custom_movements, head_lanes)
   ```

4. **Preserve turn-based assignments**: Right turns→rightmost lanes, left turns→leftmost lanes, U-turns→leftmost

## File-by-File Implementation Details

### 1. `src/args/parser.py`
**Location**: Add to `_add_network_arguments()` function around line 74

```python
parser.add_argument(
    "--custom_lanes",
    type=str,
    help="Custom lane definitions for specific edges (format: 'EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=...')"
)
```

### 2. `src/validate/validate_arguments.py`
**Location**: Add validation function and call it in `validate_arguments()` around line 44

```python
def _validate_custom_lanes(custom_lanes: str) -> None:
    """Validate custom lanes argument format and values."""
    if not custom_lanes:
        return
    
    # Parse and validate format: EdgeID=tail:N,head:ToEdge:N;EdgeID2=...
    # Validate edge ID formats (A1B1 pattern)
    # Validate tail:N and head: sections
    # Validate lane counts (1-3 range)
    # Handle dead-end syntax (head:)

def _validate_custom_lanes_file(custom_lanes_file: str) -> None:
    """Validate custom lanes file argument."""
    if not custom_lanes_file:
        return
    
    # Check file existence and readability
    # Validate file content using same logic as _validate_custom_lanes
    # Provide line-specific error messages
```

## Validation System Integration

### Complete Validation Implementation

#### 1. Add Validation Functions to `validate_arguments.py`

```python
def _validate_custom_lanes(custom_lanes: str) -> None:
    """Validate custom lanes argument format and values."""
    if not custom_lanes:
        return
    
    # Split by semicolon to get individual edge configurations
    edge_configs = [config.strip() for config in custom_lanes.split(';') if config.strip()]
    
    for config in edge_configs:
        _validate_single_edge_config(config)

def _validate_custom_lanes_file(custom_lanes_file: str) -> None:
    """Validate custom lanes file argument."""
    if not custom_lanes_file:
        return
    
    # Check file existence
    file_path = Path(custom_lanes_file)
    if not file_path.exists():
        raise ValidationError(f"Custom lanes file does not exist: {custom_lanes_file}")
    
    # Check file readability
    if not file_path.is_file():
        raise ValidationError(f"Custom lanes path is not a file: {custom_lanes_file}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except PermissionError:
        raise ValidationError(f"Cannot read custom lanes file: {custom_lanes_file}")
    except UnicodeDecodeError:
        raise ValidationError(f"Custom lanes file must be UTF-8 encoded: {custom_lanes_file}")
    except Exception as e:
        raise ValidationError(f"Error reading custom lanes file {custom_lanes_file}: {e}")
    
    # Validate file content line by line
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        try:
            _validate_single_edge_config(line)
        except ValidationError as e:
            raise ValidationError(f"Invalid custom lanes syntax on line {line_num}: {e}")

def _validate_single_edge_config(config: str) -> None:
    """Validate a single edge configuration string."""
    import re
    
    # Split by semicolon for multiple edges in one config
    edge_configs = [cfg.strip() for cfg in config.split(';') if cfg.strip()]
    
    for edge_config in edge_configs:
        # Pattern: EdgeID=tail:N,head:ToEdge:N,ToEdge2:N OR EdgeID=tail:N OR EdgeID=head:ToEdge:N
        if '=' not in edge_config:
            raise ValidationError(f"Missing '=' in configuration: {edge_config}")
        
        edge_id, specification = edge_config.split('=', 1)
        edge_id = edge_id.strip()
        specification = specification.strip()
        
        # Validate edge ID format (A1B1, B2C2, etc.)
        edge_pattern = re.compile(r'^[A-Z]+\d+[A-Z]+\d+$')
        if not edge_pattern.match(edge_id):
            raise ValidationError(f"Invalid edge ID format: {edge_id} - Must match pattern A1B1, B2C2, etc.")
        
        # Parse specification parts (tail: and/or head:)
        parts = [part.strip() for part in specification.split(',')]
        
        tail_found = False
        head_found = False
        
        for part in parts:
            if part.startswith('tail:'):
                if tail_found:
                    raise ValidationError(f"Duplicate tail specification in: {edge_config}")
                tail_found = True
                
                tail_value = part[5:].strip()  # Remove 'tail:'
                if not tail_value:
                    raise ValidationError(f"Empty tail specification in: {edge_config}")
                
                try:
                    tail_lanes = int(tail_value)
                    if tail_lanes < 1 or tail_lanes > 3:
                        raise ValidationError(f"Tail lanes must be 1-3, got {tail_lanes} in: {edge_config}")
                except ValueError:
                    raise ValidationError(f"Invalid tail lane count '{tail_value}' in: {edge_config}")
            
            elif part.startswith('head:'):
                if head_found:
                    raise ValidationError(f"Duplicate head specification in: {edge_config}")
                head_found = True
                
                head_value = part[5:].strip()  # Remove 'head:'
                
                # Handle dead-end case (empty head:)
                if not head_value:
                    continue  # Valid dead-end syntax
                
                # Parse movement specifications: ToEdge1:N,ToEdge2:M
                movements = [mov.strip() for mov in head_value.split(',') if mov.strip()]
                
                for movement in movements:
                    if ':' not in movement:
                        raise ValidationError(f"Invalid movement format '{movement}' - must be ToEdge:N")
                    
                    to_edge, lane_count = movement.split(':', 1)
                    to_edge = to_edge.strip()
                    lane_count = lane_count.strip()
                    
                    # Validate destination edge ID
                    if not edge_pattern.match(to_edge):
                        raise ValidationError(f"Invalid destination edge ID: {to_edge}")
                    
                    # Validate lane count
                    try:
                        lanes = int(lane_count)
                        if lanes < 1 or lanes > 3:
                            raise ValidationError(f"Movement lanes must be 1-3, got {lanes} for {to_edge}")
                    except ValueError:
                        raise ValidationError(f"Invalid lane count '{lane_count}' for movement {to_edge}")
            
            else:
                raise ValidationError(f"Invalid specification part '{part}' - must start with 'tail:' or 'head:'")
        
        # Ensure at least one specification (tail or head)
        if not tail_found and not head_found:
            raise ValidationError(f"Configuration must specify at least tail: or head: - got: {edge_config}")

# Integration point in validate_arguments() function
def validate_arguments(args) -> None:
    """
    Validate all command-line arguments for consistency and format correctness.
    
    Args:
        args: Parsed arguments from argparse
        
    Raises:
        ValidationError: If any argument is invalid
    """
    
    # Individual argument validations
    _validate_numeric_ranges(args)
    _validate_routing_strategy(args.routing_strategy)
    _validate_vehicle_types(args.vehicle_types)
    _validate_departure_pattern(args.departure_pattern)
    _validate_osm_file(args.osm_file)
    _validate_junctions_to_remove(args.junctions_to_remove)
    _validate_lane_count(args.lane_count)
    
    # NEW: Custom lanes validation
    _validate_custom_lanes(getattr(args, 'custom_lanes', None))
    _validate_custom_lanes_file(getattr(args, 'custom_lanes_file', None))
    
    # Cross-argument validations
    _validate_cross_arguments(args)
    _validate_sample_arguments(args)
    
    # NEW: Custom lanes cross-validation
    _validate_custom_lanes_cross_arguments(args)

def _validate_custom_lanes_cross_arguments(args) -> None:
    """Validate cross-argument constraints for custom lanes."""
    
    # Mutually exclusive: cannot use both --custom_lanes and --custom_lanes_file
    if getattr(args, 'custom_lanes', None) and getattr(args, 'custom_lanes_file', None):
        raise ValidationError("Cannot use both --custom_lanes and --custom_lanes_file simultaneously")
    
    # Custom lanes only work with synthetic grids (not OSM files)
    custom_lanes_provided = getattr(args, 'custom_lanes', None) or getattr(args, 'custom_lanes_file', None)
    if custom_lanes_provided and args.osm_file:
        raise ValidationError("Custom lanes are not supported with OSM files (--osm_file)")
    
    # Custom lanes override --lane_count but both can be specified
    # (custom lanes take precedence for specified edges, --lane_count for others)
```

#### 2. Add CLI Arguments to `parser.py`

```python
# In _add_network_arguments() function, add both arguments:

parser.add_argument(
    "--custom_lanes",
    type=str,
    help="Custom lane definitions for specific edges (format: 'EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=...')"
)

parser.add_argument(
    "--custom_lanes_file", 
    type=str,
    help="File containing custom lane definitions (one configuration per line, same format as --custom_lanes)"
)
```

### 3. `src/config.py`
**Location**: Add new configuration class around line 82

```python
@dataclass
class CustomLaneConfig:
    """Configuration for custom edge lane definitions"""
    edge_configs: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    @classmethod
    def parse_custom_lanes(cls, custom_lanes_str: str) -> 'CustomLaneConfig':
        """Parse custom lanes string into structured configuration."""
        # Implementation to parse and structure custom lane data
    
    def get_tail_lanes(self, edge_id: str) -> Optional[int]:
        """Get tail lane count for specific edge."""
    
    def get_movement_lanes(self, edge_id: str, to_edge: str) -> Optional[int]:
        """Get movement lane count for specific edge-to-edge movement."""
    
    def has_custom_config(self, edge_id: str) -> bool:
        """Check if edge has custom configuration."""
```

### 4. `src/network/custom_lanes.py` (NEW FILE)
**Main implementation file with shared code architecture**

```python
from src.network.split_edges_with_lanes import (
    analyze_movements_from_connections,
    calculate_movement_angles,
    assign_lanes_by_angle
)

def apply_custom_lane_configs(custom_lane_config: CustomLaneConfig) -> None:
    """Apply custom lane configurations using shared spatial logic."""
    
    # Phase 1: Load and analyze current XML files
    # Phase 2: Calculate bidirectional impact and new requirements
    # Phase 3: Update all XML files with complete deletion/regeneration
    # Phase 4: Regenerate traffic light logic with junction-wide recalculation

def _handle_full_specification(edge_id: str, config: Dict) -> None:
    """Handle A1B1=tail:2,head:B1B0:1,B1C1:2 - complete edge customization."""

def _handle_tail_only(edge_id: str, tail_lanes: int) -> None:
    """Handle A1B1=tail:2 - minimal impact, upstream connections only."""

def _handle_head_movements_only(edge_id: str, movement_config: Dict) -> None:
    """Handle A1B1=head:B1B0:1,B1C1:2 - downstream connections only."""

def _handle_dead_end(edge_id: str, tail_lanes: int) -> None:
    """Handle A1B1=tail:2,head: - create dead-end street."""

def _update_edges_file(edge_configs: Dict) -> None:
    """Update numLanes in edg.xml for tail and/or head segments."""

def _delete_all_connections_for_edge(edge_id: str) -> None:
    """Complete deletion strategy - remove all bidirectional connections."""

def _regenerate_connections_with_shared_logic(edge_id: str, config: Dict) -> None:
    """Regenerate using existing assign_lanes_by_angle() + custom overrides."""

def _update_bidirectional_traffic_lights(affected_junctions: Set[str]) -> None:
    """Regenerate traffic light logic for upstream AND downstream junctions."""

def _calculate_junction_connections(junction_id: str) -> List[Dict]:
    """Calculate all connections going into a specific junction."""

def _generate_traffic_light_state_string(connections: List[Dict]) -> str:
    """Generate traffic light state string based on connection count."""

def _redistribute_incoming_connections(edge_id: str, new_tail_lanes: int) -> None:
    """Handle lane reduction - redistribute upstream connections to fewer tail lanes."""
```

### 5. `src/pipeline/standard_pipeline.py`
**Location**: Add new step after step 3 (edge splitting) around line 85

```python
# Step 3.5: Apply Custom Lane Configurations (if provided)
if hasattr(self.args, 'custom_lanes') and self.args.custom_lanes:
    self._log_step(3.5, "Applying Custom Lane Configurations")
    self._execute_custom_lanes()

def _execute_custom_lanes(self) -> None:
    """Execute custom lane configuration application."""
    from src.config import CustomLaneConfig
    from src.network.custom_lanes import apply_custom_lane_configs
    
    custom_lane_config = CustomLaneConfig.parse_custom_lanes(self.args.custom_lanes)
    apply_custom_lane_configs(custom_lane_config)
    self.logger.info("Successfully applied custom lane configurations")
```

## Lane Reduction Strategies

### When Reducing Tail Lanes (Most Problematic Case)

#### Strategy 1: Proportional Redistribution
```python
# From 3 lanes to 2 lanes: redistribute proportionally
# Original: fromLane 0,1,2 → toLane 0,1,2
# New: fromLane 0,1,2 → toLane 0,1,1  (lane 2 traffic merged to lane 1)
```

#### Strategy 2: Rightmost Lane Preservation  
```python
# From 3 lanes to 2 lanes: keep rightmost lanes
# Original: fromLane 0,1,2 → toLane 0,1,2
# New: fromLane 0,1 → toLane 0,1  (remove lane 2 connections)
```

#### Strategy 3: Spatial Logic Preservation (RECOMMENDED)
```python
# Use existing assign_lanes_by_angle() to redistribute
# Maintain turn-based assignments even with fewer lanes
# Right turns → lane 0, straight → lane 1, left turns → lane 1 (merged)
```

## Edge Cases and Validation

### Critical Validation Requirements:
1. **Upstream Capacity Check**: Ensure incoming edges can redistribute to fewer tail lanes
2. **Movement Preservation**: All movements must remain routable after customization
3. **Junction Compatibility**: Traffic light logic must handle connection changes
4. **Flow Conservation**: Total lane capacity should not create bottlenecks
5. **Reachability**: Every destination must remain accessible

### Edge Cases to Handle:
1. **Tail Lanes > Head Lanes**: When tail-only customization creates imbalance
2. **Single Lane Movements**: When custom specification results in 0-lane movements
3. **U-turn Conflicts**: Special handling for U-turn lane assignments
4. **Dead-end Streets**: OSM compatibility with irregular topologies
5. **Minimum Lane Requirements**: Ensuring at least 1 lane per movement

## Expected Complexity and Challenges

### VERY HIGH Complexity Areas (Revised Assessment)
1. **Bidirectional Impact Management**: One edge change affects TWO junctions
2. **Junction-Wide Traffic Light Recalculation**: Complete state string regeneration
3. **Lane Redistribution Algorithms**: Handling lane reduction scenarios
4. **Spatial Logic Integration**: Preserving turn-based assignments with shared code

### HIGH Complexity Areas
1. **Complete Deletion/Regeneration Strategy**: Avoiding partial update conflicts
2. **linkIndex Recalculation**: Sequential reassignment across all affected connections
3. **Flexible Configuration Parsing**: Supporting tail-only, head-only, full specifications
4. **XML File Coordination**: Ensuring consistency across 4 XML files simultaneously

### MEDIUM Complexity Areas  
1. **Shared Code Architecture**: Reusing existing functions without duplication
2. **Movement-to-Lane Assignment**: Applying custom specs within spatial constraints
3. **Validation System**: Format, existence, and flow conservation checking

### LOW Complexity Areas
1. **CLI Argument Parsing**: Standard argparse implementation
2. **Configuration Management**: Dataclass-based structure
3. **Basic Format Validation**: Edge ID patterns and lane count ranges

## Testing Strategy

### Unit Tests
1. Custom lane parsing and validation
2. Configuration object management
3. Individual XML file updates

### Integration Tests  
1. End-to-end custom lane application
2. Traffic light logic consistency
3. Connection flow validation

### Validation Tests
1. SUMO network validation after custom changes
2. Junction traffic light state verification
3. Movement preservation testing

## Example Usage

### Full Specification Examples
```bash
# Complete edge customization with explicit head: section
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2,head:B1B0:1,B1C1:2" \
  --gui

# Multiple edge specifications
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2,head:B1B0:1,B1C1:2;B2C2=tail:3,head:C2C1:2,C2D2:1" \
  --gui
```

### Tail-Only Examples (Preserve All Movements)
```bash
# Only modify tail lanes - preserve all existing head movements
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2;A2B2=tail:3;B1C1=tail:1" \
  --gui
```

### Head Movements Only Examples
```bash
# Only customize head movements - keep automatic tail lanes
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=head:B1B0:1,B1C1:2;A2B2=head:B2C2:2,B2B1:1" \
  --gui
```

### Dead-End Examples
```bash
# Create dead-end streets (vehicles can enter but not exit)
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2,head:;A2B2=tail:3,head:" \
  --gui
```

### Mixed Approach Examples
```bash
# Different customization types for different edges
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2;A2B2=head:B2C2:2,B2B1:1;B1C1=tail:3,head:C1D1:2;C1D1=head:" \
  --gui
```

## Documentation Strategy

### SPECIFICATION.md Updates

The following content should be added to SPECIFICATION.md to document the custom edge lane definition feature:

```markdown
## Custom Edge Lane Definition System

### Overview
The system supports manual specification of lane configurations for specific edges in synthetic grid networks, overriding automatic lane assignment algorithms. This is implemented as a post-processing step after the standard edge splitting with flow-based lane assignment.

### Feature Specification

#### CLI Arguments
- `--custom_lanes`: Direct specification of custom lane configurations
  - Format: `"EdgeID=tail:N,head:ToEdge1:N,ToEdge2:N;EdgeID2=..."`
  - Supports flexible syntax: tail-only, head-only, full specification, dead-end creation
  - Multiple edge configurations separated by semicolons

- `--custom_lanes_file`: File-based configuration for complex scenarios
  - Same syntax as CLI argument, one configuration per line
  - Supports comments (lines starting with #) and empty lines
  - UTF-8 encoding required
  - Mutually exclusive with `--custom_lanes`

#### Supported Configuration Types
1. **Full Specification**: `A1B1=tail:2,head:B1B0:1,B1C1:2`
   - Custom tail lanes + explicit head movement assignments
   - Complete edge reconfiguration with bidirectional impact

2. **Tail-Only Customization**: `A1B1=tail:2`
   - Custom tail lanes, preserve all existing head movements
   - Minimal impact scope, only affects upstream junction

3. **Head Movements Only**: `A1B1=head:B1B0:1,B1C1:2`
   - Automatic tail lanes, custom movement assignments
   - Only affects downstream junction

4. **Dead-End Creation**: `A1B1=tail:2,head:`
   - Custom tail lanes, remove ALL head movements
   - Creates dead-end street (vehicles can enter but not exit)

#### Technical Implementation
- **Integration Point**: Step 3.5 in pipeline (after edge splitting, before traffic generation)
- **Shared Code Architecture**: Reuses existing spatial logic functions from `split_edges_with_lanes.py`
- **Complete Deletion/Regeneration Strategy**: Avoids partial update conflicts
- **Bidirectional Impact Management**: Updates both upstream and downstream junctions
- **Traffic Light System Regeneration**: Complete recalculation of state strings and linkIndex values

#### Constraints and Limitations
- **Synthetic Networks Only**: Not supported with OSM files (`--osm_file`)
- **Grid Network Requirement**: Edge IDs must follow pattern `A1B1`, `B2C2`, etc.
- **Lane Count Bounds**: Tail and movement lane counts must be 1-3
- **Movement Selection**: Only user-specified movements exist, others are deleted
- **Precedence**: Overrides `--lane_count` for specified edges

#### Validation Rules
- Edge ID format validation (must match grid pattern)
- Lane count bounds checking (1-3 range)
- Movement destination validation
- File existence and readability (for file-based config)
- Mutual exclusivity between CLI and file options
- Cross-argument compatibility checking

#### Error Handling
- Immediate exit on validation errors with descriptive messages
- Line-specific error reporting for file-based configurations
- Network validation using SUMO's built-in tools
- No automatic error recovery or rollback mechanisms

#### Use Cases
- **Research Applications**: Test specific lane configurations for algorithm validation
- **Scenario Testing**: Create controlled traffic conditions for Tree Method evaluation
- **Construction Simulation**: Block or restrict specific routes
- **Intersection Modeling**: Match real-world intersection configurations
- **Edge Case Testing**: Validate system behavior under unusual conditions
```

### Integration Notes
This specification should be added to the "Advanced Features" section of SPECIFICATION.md, maintaining consistency with existing documentation style and cross-referencing related features like `--lane_count` and traffic control methods.

## Benefits and Integration

### Benefits
1. **Precise Experimental Control**: Researchers can test specific lane configurations
2. **Scenario Testing**: Validate Tree Method performance under custom constraints  
3. **Real-World Modeling**: Match specific intersection configurations
4. **Algorithmic Validation**: Test edge cases and boundary conditions

### Integration
1. **Maintains Compatibility**: All existing functionality unchanged
2. **Post-Processing Approach**: Clean separation from core edge splitting logic
3. **Validation Integration**: Hooks into existing pipeline validation
4. **Error Handling**: Comprehensive error messages and recovery

This comprehensive plan provides complete coverage for implementing custom edge lane definitions while maintaining the integrity and functionality of the existing traffic generation system.