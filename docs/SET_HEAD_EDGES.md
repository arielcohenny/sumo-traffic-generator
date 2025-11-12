# Multi-Head Edge Architecture: Implementation Roadmap

## Overview

Transform edge splitting from **1 tail → 1 head edge** to **1 tail → up to 4 head edges** (one per movement type: right, straight, left, u-turn).

**Key Insight**: Keep existing lane assignment logic intact (it works well), then post-process the single head edge to split it into multiple movement-specific head edges.

## Architecture Change

### Current System
```
Tail Edge (A0B0)
    ↓
Head Edge (A0B0_H_s) [lanes 0,1,2,3,4]
    ↓
Lane 0 → Right turn
Lane 1 → Right turn
Lane 2 → Straight
Lane 3 → Left turn
Lane 4 → U-turn
```

### Proposed System
```
Tail Edge (A0B0)
    ↓
├─ A0B0_H_right [lanes 0,1] → Right turn only
├─ A0B0_H_straight [lane 0] → Straight only
├─ A0B0_H_left [lane 0] → Left turn only
└─ A0B0_H_uturn [lane 0] → U-turn only
```

## Implementation Strategy

### Phase 1: Post-Processing Logic (NEW FUNCTION)

**File**: `src/network/split_edges_with_lanes.py`

**Add new function**: `split_head_by_movements()`
- **Input**: Existing head edge with lane-to-movement mappings
- **Process**:
  1. Group lanes by movement type (right, straight, left, uturn)
  2. Create separate head edge for each movement type present
  3. Renumber lanes within each new head edge (start from 0)
  4. Update intermediate node connections
- **Output**: Dictionary of movement-specific head edges

**Integration point**: Call after `split_edges_at_head_distance()` completes

### Phase 2: Connection Updates

**File**: `src/network/split_edges_with_lanes.py`

**Modify**: `update_connections_file()` (lines 437-551)

1. **External connections** (heads → downstream):
   - Each movement-specific head edge connects to its ONE destination
   - Map old head lane indices to new head edge + lane combinations

2. **Internal connections** (tail → heads):
   - Distribute tail lanes across multiple head edges
   - Maintain proportional distribution based on movement demands

### Phase 3: Traffic Light Strategy Adaptation

**File**: `src/network/generate_grid.py`

**Update all three strategies**:

1. **Opposites Strategy** (lines 274-353):
   - North phase includes: `_H_right`, `_H_straight`, `_H_left`, `_H_uturn` from north edges
   - South phase includes: `_H_right`, `_H_straight`, `_H_left`, `_H_uturn` from south edges

2. **Incoming Strategy**:
   - Each edge's phase includes all its movement-specific head edges

3. **Partial Opposites Strategy** (lines 356-590):
   - Phase 1: `_H_right` + `_H_straight` edges
   - Phase 2: `_H_left` + `_H_uturn` edges

**Helper function**: `collect_head_edges_for_base_edge(base_edge_id)` → returns list of all head edges

### Phase 4: Traffic Light File Updates

**File**: `src/network/split_edges_with_lanes.py`

**Modify**: `update_traffic_lights_file()` (lines 553-575)
- Update connection references to use new head edge naming
- Map old single-head structure to multi-head structure
- Ensure all head edges are included in appropriate phases

### Phase 5: Tree Method Integration

**File**: `src/traffic_control/decentralized_traffic_bottlenecks/shared/classes/net_data_builder.py`

**Update Edge class** (lines 60-84):
1. Modify pattern matching to recognize: `_H_right`, `_H_straight`, `_H_left`, `_H_uturn`
2. Store all head edges per base edge (not just one)
3. Ensure algorithm handles list of heads correctly

**Test**: Verify Tree Method's bottleneck detection works with multiple heads

### Phase 6: Validation Updates

**File**: `src/validate/validate_split_edges_with_lanes.py`

**Remove**:
- Check that "head lanes have exactly ONE outgoing direction" (line 248)

**Add new validations**:
- Each head edge serves only one movement type
- All movement types present in original analysis have corresponding head edges
- Lane counts sum correctly across all head edges
- Tail connects to all required head edges

### Phase 7: Data Structure Changes

**Update `split_edges` dictionary structure**:

```python
# Current:
split_edges[edge_id] = {
    'tail': tail_segment,
    'head': head_segment,  # Single head
    'tail_lanes': int,
    'head_lanes': int,
    ...
}

# Proposed:
split_edges[edge_id] = {
    'tail': tail_segment,
    'heads': {  # Multiple heads by movement type
        'right': {
            'edge': head_segment,
            'lanes': int,
            'original_lanes': [0, 1]  # Original lane indices
        },
        'straight': {...},
        'left': {...},
        'uturn': {...}
    },
    'tail_lanes': int,
    'total_head_lanes': int,
    ...
}
```

## Detailed Implementation Steps

### Step 1: Create `split_head_by_movements()` function

**Location**: After `split_edges_at_head_distance()` in `split_edges_with_lanes.py`

**Inputs**:
- Original head edge definition
- Lane assignments from `assign_lanes_by_angle()`
- Movement analysis data

**Algorithm**:
```python
def split_head_by_movements(head_edge, lane_assignments, movements_data):
    """
    Split single head edge into multiple movement-specific head edges.

    Returns:
        dict: {
            'right': {'edge': edge_def, 'lanes': [...], 'to_edge': 'X'},
            'straight': {...},
            'left': {...},
            'uturn': {...}
        }
    """
    # 1. Group lanes by movement type
    movement_groups = group_lanes_by_movement(lane_assignments)

    # 2. Create head edge for each movement type
    head_edges = {}
    for movement_type, lane_list in movement_groups.items():
        new_edge_id = f"{base_edge_id}_H_{movement_type}"
        new_edge = create_edge_definition(
            edge_id=new_edge_id,
            from_node=intermediate_node,
            to_node=downstream_junction,
            num_lanes=len(lane_list),
            geometry=head_geometry
        )
        head_edges[movement_type] = {
            'edge': new_edge,
            'lanes': lane_list,
            'to_edge': get_destination_edge(movement_type, movements_data)
        }

    return head_edges
```

### Step 2: Integrate into main pipeline

**Modify**: `split_edges_with_flow_based_lanes()` (lines 26-81)

```python
# After line 66 (split_edges_at_head_distance completes):
split_edges = split_edges_at_head_distance(...)

# NEW: Post-process heads to split by movement
for edge_id in split_edges:
    head_data = split_edges[edge_id]
    movement_analysis = movements_by_edge[edge_id]
    lane_assignments = head_data['lane_assignments']

    # Split single head into multiple heads
    multi_heads = split_head_by_movements(
        head_edge=head_data['head'],
        lane_assignments=lane_assignments,
        movements_data=movement_analysis
    )

    # Replace single head with multi-head structure
    split_edges[edge_id]['heads'] = multi_heads
    del split_edges[edge_id]['head']  # Remove old single head
```

### Step 3: Update XML writers

**Nodes file** (`.nod.xml`): No changes needed (intermediate node stays same)

**Edges file** (`.edg.xml`):
- Remove single head edge
- Add multiple head edges (one per movement type)

**Connections file** (`.con.xml`):
- Update external connections: Each head edge → its destination
- Update internal connections: Tail → all head edges

**Traffic lights file** (`.tll.xml`):
- Update phase connections to reference multiple head edges

### Step 4: Update traffic light strategies

**Helper function** (add to `generate_grid.py`):
```python
def get_head_edges_for_base_edge(edge_id, edge_file):
    """
    Find all head edges for a given base edge.

    Returns: ['A0B0_H_right', 'A0B0_H_straight', 'A0B0_H_left', 'A0B0_H_uturn']
    """
    pattern = f"{edge_id}_H_"
    head_edges = [e['@id'] for e in edges if e['@id'].startswith(pattern)]
    return head_edges
```

**Modify strategy functions**:
- Instead of referencing single head edge, collect all head edges
- Include all in appropriate phases

### Step 5: Update Tree Method adapter

**Modify**: `net_data_builder.py`

```python
# Current (line 60-84):
is_head = True if '_H_' in edge['@id'] else False
first_part = edge['@id'].split('_H_')[0]

# Proposed:
is_head = '_H_' in edge['@id']
if is_head:
    # Extract base edge and movement type
    base_edge_id = edge['@id'].split('_H_')[0]
    movement_type = edge['@id'].split('_H_')[1]  # right, straight, left, uturn

    # Add to edge object's heads list
    edge_obj.heads.append(edge['@id'])
```

## Testing Strategy

### 1. Unit Tests
- Test `split_head_by_movements()` with various lane configurations
- Test movement type classification

### 2. Integration Testing
```bash
# Small 3x3 grid with GUI
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --end-time 1800 --gui

# Verify in SUMO GUI:
# - Multiple head edges visible per junction approach
# - Lanes correctly assigned to edges
# - Traffic lights control all head edges
```

### 3. Validation Testing
```bash
# Run with validation enabled
# Ensure all new validation checks pass
```

### 4. Traffic Control Method Testing
```bash
# Test all three methods with new structure
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --end-time 3600 --traffic_control tree_method --gui
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --end-time 3600 --traffic_control actuated --gui
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 5 --num_vehicles 500 --end-time 3600 --traffic_control fixed --gui
```

### 5. Visual Verification
- Open SUMO GUI and examine junction structures
- Verify separate head edges for each movement type
- Check lane connections are correct
- Observe traffic light phases controlling multiple head edges

## Risk Mitigation

### Low-Risk Areas (No Changes)
- Traffic generation (uses base edges only)
- Edge attractiveness system
- Zone extraction
- Vehicle types and routing strategies

### Medium-Risk Areas (Careful Testing Required)
- Traffic light phase timing (may need adjustment)
- Tree Method algorithm (must handle multiple heads)
- Connection distribution (tail → multiple heads)

### High-Risk Areas (Thorough Validation Required)
- Lane indexing across multiple head edges
- Traffic light phase structure
- Network topology consistency

## Rollback Strategy

If issues arise:
1. Keep original `split_edges_with_lanes.py` as `split_edges_with_lanes_backup.py`
2. Add feature flag to enable/disable multi-head splitting
3. Can revert to single-head structure without data loss

## Success Criteria

- [ ] All edges successfully split into movement-specific head edges
- [ ] Traffic lights control all head edges correctly
- [ ] Tree Method algorithm works with new structure
- [ ] All validation checks pass
- [ ] SUMO simulation runs without errors
- [ ] Visual inspection confirms correct lane assignments
- [ ] Performance comparable to current system

## Timeline Estimate

- Phase 1-2 (Core splitting + connections): 4-6 hours
- Phase 3-4 (Traffic lights): 3-4 hours
- Phase 5 (Tree Method): 2-3 hours
- Phase 6-7 (Validation + data structures): 2-3 hours
- Testing and debugging: 4-6 hours
- **Total**: 15-22 hours

---

## Next Steps

Once this roadmap is approved:
1. Create backup of critical files
2. Implement Phase 1 (core splitting logic)
3. Test incrementally after each phase
4. Update documentation as changes are made

---

# The New Approach: Let SUMO Handle Geometry

## Core Principle

Instead of manually calculating geometry offsets with hardcoded values, **let SUMO's netconvert automatically position parallel edges** - just like it already does for multiple lanes within a single edge.

**Key Insight**: If SUMO can auto-position 3 lanes within a single edge correctly, it should auto-position 3 edges between the same nodes correctly. Same logic.

## Implementation

### 1. Define Multiple Head Edges (NO shape attributes)

```xml
<!-- In .edg.xml -->
<edge id="A1B1" from="A1" to="A1B1_H_node" numLanes="3" />

<!-- Multiple heads - same from/to nodes, NO shape specified -->
<edge id="A1B1_H_right" from="A1B1_H_node" to="B1" numLanes="1" />
<edge id="A1B1_H_straight" from="A1B1_H_node" to="B1" numLanes="1" />
<edge id="A1B1_H_left" from="A1B1_H_node" to="B1" numLanes="1" />
<!-- Only create head edges for movements that exist -->
```

**Important**:
- All head edges have the **same from/to nodes**
- **DO NOT specify shape attributes** - let SUMO calculate them
- Only include edges for movements that actually exist (not always all 4)

### 2. Define Connections (specify lane-to-edge mappings)

```xml
<!-- In .con.xml -->
<!-- Tail to heads: Map which tail lanes feed which head edges -->
<connection from="A1B1" to="A1B1_H_right" fromLane="0" toLane="0" />
<connection from="A1B1" to="A1B1_H_straight" fromLane="1" toLane="0" />
<connection from="A1B1" to="A1B1_H_left" fromLane="2" toLane="0" />

<!-- Heads to destinations: Each head connects to its specific destination -->
<connection from="A1B1_H_right" to="B1C1" fromLane="0" toLane="0" />
<connection from="A1B1_H_straight" to="B1B2" fromLane="0" toLane="0" />
<connection from="A1B1_H_left" to="B1A1" fromLane="0" toLane="0" />
```

**Connection Mapping**:
- **Internal connections** (tail → heads): Specify which tail lanes feed which head edges
- **External connections** (heads → destinations): Each head edge goes to its movement-specific destination
- **Lane renumbering**: Each head edge starts with lane 0 (toLane indices are renumbered)

### 3. Let netconvert do the rest

Netconvert will automatically:
- ✅ Position parallel edges with proper lateral spacing
- ✅ Handle opposite direction separation
- ✅ Calculate all geometry and shapes
- ✅ Assign traffic light linkIndices correctly
- ✅ Avoid overlaps between edges

## Why This Works

**SUMO already does this for lanes within edges:**
- Single edge with 3 lanes → SUMO spreads them laterally with 3.2m lane width spacing
- Multiple edges between same nodes → SUMO applies the same logic for edge separation

**No hardcoded geometry calculations needed** - SUMO knows how to space parallel road elements.

## Implementation Steps

### 1. Remove Geometry Calculation Code
- **Delete** `calculate_offset_shape()` function (hardcoded 1.6m values are incorrect)
- **Delete** `classify_movement_type()` helper (can be recreated simpler if needed)
- **Remove** any manual shape attribute generation

### 2. Modify `post_process_to_multi_head_edges()`
- Group lanes by movement type (keep this logic)
- Create head edge definitions **WITHOUT shape attributes**
- Store only: `id`, `from`, `to`, `numLanes`, `speed`, `priority`
- **Do not** calculate or store shape coordinates

### 3. Update `update_edges_file()`
- Write multiple head edges (only those that exist for each base edge)
- **Do NOT include `shape` attribute** in edge XML elements
- Let netconvert generate shapes automatically

### 4. Update `update_connections_file()`
- Remap tail→head connections to use movement-specific head edges
- Renumber `toLane` indices (each head edge starts at lane 0)
- Remap head→destination connections with updated `from` edge IDs and `fromLane` indices

### 5. Test with 3x3 Grid
```bash
env PYTHONUNBUFFERED=1 python -m src.cli --grid_dimension 3 --num_vehicles 100 --end-time 1800 --gui
```

- Let SUMO auto-position everything
- Verify in GUI that edges are properly separated
- Check that opposite directions don't overlap
- Confirm traffic lights control all head edges

## Benefits of This Approach

1. **No hardcoded values** - eliminates magic numbers like 1.6m
2. **Works for any lane configuration** - SUMO adapts to different lane counts
3. **Simpler code** - less geometry calculation, fewer functions
4. **Proven approach** - relies on SUMO's built-in capabilities
5. **Maintainable** - easier to understand and modify

## What Gets Deleted

Files/functions that are no longer needed:
- `calculate_offset_shape()` function in `split_edges_with_lanes.py`
- Manual shape coordinate calculations
- Hardcoded offset constants (1.6m, etc.)

## Success Criteria

- [ ] Multiple head edges created for each movement type
- [ ] No shape attributes in edge definitions
- [ ] netconvert runs successfully without geometry errors
- [ ] SUMO GUI shows properly separated parallel edges
- [ ] Opposite directions don't overlap
- [ ] Traffic lights control all head edges correctly
- [ ] Simulation runs without errors

## Estimated Timeline

- Remove geometry code: 30 minutes
- Modify `post_process_to_multi_head_edges()`: 1 hour
- Update `update_edges_file()`: 30 minutes
- Update `update_connections_file()`: 1-2 hours
- Testing and verification: 1-2 hours
- **Total**: 4-6 hours
