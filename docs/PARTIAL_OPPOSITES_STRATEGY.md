# Partial Opposites Traffic Light Strategy - Implementation Roadmap

## Overview

This document provides a technical implementation roadmap for adding the `partial_opposites` traffic light strategy to the SUMO traffic generator.

### Strategy Description

The `partial_opposites` strategy implements a 4-phase traffic signal system:

1. **Phase 1**: North/South straight + right turns (green)
2. **Phase 2**: North/South left turns + U-turns (green)
3. **Phase 3**: East/West straight + right turns (green)
4. **Phase 4**: East/West left turns + U-turns (green)

Each green phase includes a yellow transition phase (3 seconds), resulting in 8 total phases per cycle.

### Key Characteristics

- **Cycle Time**: 90 seconds (same as `opposites` strategy)
  - Straight+right: 30s green + 3s yellow = 33s per direction
  - Left+u-turn: 9s green + 3s yellow = 12s per direction
  - Total per direction (N/S or E/W): 45s
- **Minimum Lane Requirement**: 2 lanes per edge (one for straight+right, one for left+u-turn)
- **Movement Separation**: Reduces conflicts between through traffic and turning movements
- **Backward Compatibility**: NO impact on existing `opposites` and `incoming` strategies

---

## Minimum Lane Requirement

### Why 2 Lanes Are Required

With `partial_opposites`, movements are separated into distinct phases:
- **Phase group 1**: Straight + Right movements
- **Phase group 2**: Left + U-turn movements

If an edge has only 1 lane:
- A left-turning vehicle at the front blocks ALL vehicles behind it
- Straight/right vehicles cannot use their dedicated green phase
- **Result**: The strategy's benefits are completely negated

**Minimum Configuration**:
- **Lane 0 (rightmost)**: Handles straight + right turn movements
- **Lane 1 (leftmost)**: Handles left turn + U-turn movements

**Better Configuration** (3 lanes):
- **Lane 0 (rightmost)**: Right turn movements
- **Lane 1 (middle)**: Straight movements
- **Lane 2 (leftmost)**: Left turn + U-turn movements

---

## Implementation Checklist

### Phase 1: Foundation Changes (Low Risk)
- [ ] Add constants to `src/constants.py`
- [ ] Update CLI arguments in `src/args/parser.py`
- [ ] Add validation in `src/validate/validate_arguments.py`

### Phase 2: Lane Count Enforcement (Medium Risk)
- [ ] Modify `realistic` algorithm in `src/network/lane_counts.py`
- [ ] Modify `random` algorithm in `src/network/lane_counts.py`
- [ ] Update function signatures to propagate `traffic_light_strategy`
- [ ] **CRITICAL**: Test that `opposites` and `incoming` are unaffected

### Phase 3: Traffic Light Conversion (High Risk)
- [ ] Create `convert_to_partial_opposites_strategy()` in `src/network/generate_grid.py`
- [ ] Implement movement classification logic
- [ ] Generate 8-phase traffic light structure
- [ ] Add integration point in `generate_grid_network()`

### Phase 4: Validation Updates (Medium Risk)
- [ ] Update lane count validation in `src/validate/validate_network.py`
- [ ] Add traffic light phase validation for `partial_opposites`

### Phase 5: Testing (Critical)
- [ ] Test 3x3 grid with `partial_opposites`
- [ ] Regression test: verify `opposites` unchanged
- [ ] Regression test: verify `incoming` unchanged
- [ ] Test with all traffic control methods (tree_method, actuated, fixed, rl)
- [ ] Test edge cases (fixed 1 lane = error, fixed 2 lanes = works)
- [ ] Full integration test with 5x5 grid

### Phase 6: Documentation
- [ ] Update `CLAUDE.md` with new strategy examples
- [ ] Update CLI help text

---

## Detailed Implementation Steps

### Step 1: Add Constants (`src/constants.py`)

Add after line 183 (after `DEFAULT_TRAFFIC_LIGHT_STRATEGY`):

```python
# Traffic Light Strategy Constants
TL_STRATEGY_OPPOSITES = 'opposites'
TL_STRATEGY_INCOMING = 'incoming'
TL_STRATEGY_PARTIAL_OPPOSITES = 'partial_opposites'

# Minimum Lanes by Traffic Light Strategy
MIN_LANES_FOR_TL_STRATEGY = {
    TL_STRATEGY_OPPOSITES: 1,
    TL_STRATEGY_INCOMING: 1,
    TL_STRATEGY_PARTIAL_OPPOSITES: 2,  # Requires 2+ lanes
}

# Partial Opposites Phase Durations
PARTIAL_OPPOSITES_STRAIGHT_RIGHT_GREEN = 30  # seconds
PARTIAL_OPPOSITES_LEFT_UTURN_GREEN = 9       # seconds
PARTIAL_OPPOSITES_YELLOW = 3                 # seconds
# Total cycle: (30+3) + (9+3) = 45s per direction × 2 directions = 90s
```

**Risk**: Low - Adding new constants doesn't affect existing code

---

### Step 2: Update CLI Arguments (`src/args/parser.py`)

Modify line 213 (the `choices` parameter):

```python
# OLD:
choices=["opposites", "incoming"],

# NEW:
choices=["opposites", "incoming", "partial_opposites"],
```

Update help text (line 215):

```python
# OLD:
help=f"Traffic light phasing strategy: '{DEFAULT_TRAFFIC_LIGHT_STRATEGY}' (default, opposing directions together) or 'incoming' (each edge gets own phase)"

# NEW:
help=f"Traffic light phasing strategy: '{DEFAULT_TRAFFIC_LIGHT_STRATEGY}' (default, opposing directions together), 'incoming' (each edge gets own phase), or 'partial_opposites' (straight+right and left+u-turn in separate phases, requires 2+ lanes)"
```

**Risk**: Low - Adding to choices list doesn't break existing usage

---

### Step 3: Add Validation (`src/validate/validate_arguments.py`)

Add new validation function (insert near other validation functions):

```python
def validate_traffic_light_lane_compatibility(args):
    """Validate lane count is compatible with traffic light strategy

    partial_opposites strategy requires minimum 2 lanes per edge to separate:
    - Lane 0 (rightmost): straight + right movements
    - Lane 1 (leftmost): left + u-turn movements
    """
    if args.traffic_light_strategy == "partial_opposites":
        # Check if lane assignment is disabled (should not happen with defaults)
        if args.lane_count == "0":
            raise ValidationError(
                "partial_opposites strategy requires lane assignment. "
                "Cannot use --lane_count 0"
            )

        # Check for explicit fixed 1-lane configuration
        if args.lane_count.startswith("fixed"):
            try:
                lane_value = int(args.lane_count.split()[1]) if " " in args.lane_count else int(args.lane_count.replace("fixed", ""))
                if lane_value < 2:
                    raise ValidationError(
                        f"partial_opposites strategy requires minimum 2 lanes per edge. "
                        f"You specified: {args.lane_count}. "
                        f"Use '--lane_count fixed 2' or higher, or use 'realistic'/'random' algorithms."
                    )
            except (ValueError, IndexError):
                # If parsing fails, let other validation catch it
                pass
```

Call this function in the main validation flow (find where other validation functions are called and add):

```python
validate_traffic_light_lane_compatibility(args)
```

**Risk**: Low - Validation only runs at startup, fails fast with clear error message

---

### Step 4: Modify Lane Count Algorithms (`src/network/lane_counts.py`)

#### 4.1. Update `calculate_lane_count()` Function Signature

Add `traffic_light_strategy` parameter (around line 47):

```python
# OLD:
def calculate_lane_count(algorithm: str, edge_id: str, demand_score: float = None,
                         zones_data=None, rng=None) -> int:

# NEW:
def calculate_lane_count(algorithm: str, edge_id: str, demand_score: float = None,
                         zones_data=None, rng=None, traffic_light_strategy: str = "opposites") -> int:
```

#### 4.2. Modify Realistic Algorithm (lines 59-67)

```python
# OLD:
if algorithm == "realistic":
    if demand_score is None:
        raise ValueError("realistic algorithm requires demand_score")

    if demand_score < 1.0:
        lanes = 1
    elif demand_score < 2.0:
        lanes = 2
    else:
        lanes = 3

# NEW:
if algorithm == "realistic":
    if demand_score is None:
        raise ValueError("realistic algorithm requires demand_score")

    # Enforce minimum based on traffic light strategy
    min_lanes = MIN_LANES_FOR_TL_STRATEGY.get(traffic_light_strategy, 1)

    if demand_score < 1.0:
        lanes = max(min_lanes, 1)
    elif demand_score < 2.0:
        lanes = max(min_lanes, 2)
    else:
        lanes = 3
```

#### 4.3. Modify Random Algorithm (line 73)

```python
# OLD:
elif algorithm == "random":
    if rng is None:
        raise ValueError("random algorithm requires rng")
    lanes = rng.randint(min_lanes, max_lanes)

# NEW:
elif algorithm == "random":
    if rng is None:
        raise ValueError("random algorithm requires rng")
    # Enforce minimum based on traffic light strategy
    strategy_min_lanes = MIN_LANES_FOR_TL_STRATEGY.get(traffic_light_strategy, 1)
    effective_min = max(min_lanes, strategy_min_lanes)
    lanes = rng.randint(effective_min, max_lanes)
```

Add import at top of file:

```python
from src.constants import MIN_LANES_FOR_TL_STRATEGY
```

**Risk**: Medium - Changes core lane calculation logic
**Mitigation**: Only affects lanes when `traffic_light_strategy == "partial_opposites"`

---

### Step 5: Propagate `traffic_light_strategy` Parameter

#### 5.1. Update `split_edges_with_flow_based_lanes()` (`src/network/split_edges_with_lanes.py`)

Add parameter to function signature (around line 323):

```python
# OLD:
def split_edges_with_flow_based_lanes(lane_count_arg, junction_radius=10.0,
                                      zones_data=None, seed=None):

# NEW:
def split_edges_with_flow_based_lanes(lane_count_arg, junction_radius=10.0,
                                      zones_data=None, seed=None,
                                      traffic_light_strategy="opposites"):
```

Pass to `calculate_lane_count()` calls (around line 345):

```python
# OLD:
tail_lanes = calculate_lane_count(algorithm, edge_id, demand_score, zones_data, rng)

# NEW:
tail_lanes = calculate_lane_count(algorithm, edge_id, demand_score, zones_data, rng,
                                   traffic_light_strategy=traffic_light_strategy)
```

#### 5.2. Update Call Site in `generate_grid.py`

Find where `split_edges_with_flow_based_lanes()` is called (around line 370-380) and add parameter:

```python
# OLD:
split_edges_with_flow_based_lanes(lane_count_arg, junction_radius, zones_data, seed)

# NEW:
split_edges_with_flow_based_lanes(lane_count_arg, junction_radius, zones_data, seed,
                                   traffic_light_strategy=traffic_light_strategy)
```

**Risk**: Low - Just threading parameters through
**Testing**: Verify `opposites` and `incoming` still produce same lane counts as before

---

### Step 6: Create Traffic Light Conversion Function (`src/network/generate_grid.py`)

Add new function after `convert_to_incoming_strategy()` (after line 356):

```python
def convert_to_partial_opposites_strategy():
    """Convert traffic light layout to partial_opposites strategy

    Creates 4-phase system where:
    - Phase 1: N/S straight + right turns (30s green)
    - Phase 2: N/S left turns + U-turns (9s green)
    - Phase 3: E/W straight + right turns (30s green)
    - Phase 4: E/W left turns + U-turns (9s green)

    Each green phase includes a yellow transition (3s), resulting in 8 phases total.
    Cycle time: 90 seconds (same as opposites strategy)

    Movement classification:
    - Calculates turn angles from edge directions
    - Uses geometry of incoming/outgoing edges
    - Straight+right: angles from -90° to +10°
    - Left+u-turn: angles from +10° to +180°

    Direction classification:
    - Parses junction IDs from edge names to determine edge direction
    - Uses junction positions to calculate which way edge is heading (N/S/E/W)
    - Groups opposing directions (N/S vs E/W) for phasing

    Requires: Minimum 2 lanes per edge to separate movement groups
    """
    import xml.etree.ElementTree as ET
    from src.constants import (
        PARTIAL_OPPOSITES_STRAIGHT_RIGHT_GREEN,
        PARTIAL_OPPOSITES_LEFT_UTURN_GREEN,
        PARTIAL_OPPOSITES_YELLOW
    )

    print(f"Converting traffic lights to partial_opposites strategy...")

    # Parse the traffic light logic file
    tree = ET.parse(CONFIG.network_tll_file)
    root = tree.getroot()

    # Parse network connection file to get movement information
    conn_tree = ET.parse(CONFIG.network_con_file)
    conn_root = conn_tree.getroot()

    # Build connection database with movement classification
    connections_db = {}  # tl_id -> list of connections with metadata

    for tl_logic in root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        connections_db[tl_id] = []

        # Find all connections controlled by this traffic light
        for connection in conn_root.findall('connection'):
            if connection.get('tl') == tl_id:
                from_edge = connection.get('from')
                to_edge = connection.get('to')
                from_lane = connection.get('fromLane')
                to_lane = connection.get('toLane')
                link_index = int(connection.get('linkIndex', -1))

                # Determine edge directions from junction IDs
                from_direction = get_edge_direction(from_edge)
                to_direction = get_edge_direction(to_edge)

                # Calculate turn angle
                turn_angle = calculate_turn_angle(from_direction, to_direction)

                # Classify movement type based on turn angle
                movement_type = classify_movement(turn_angle)

                # Determine approach orientation (N/S or E/W)
                approach_orientation = get_edge_orientation(from_edge)

                connections_db[tl_id].append({
                    'from': from_edge,
                    'to': to_edge,
                    'from_lane': from_lane,
                    'to_lane': to_lane,
                    'link_index': link_index,
                    'from_direction': from_direction,     # N, S, E, or W
                    'to_direction': to_direction,         # N, S, E, or W
                    'turn_angle': turn_angle,             # degrees
                    'movement_type': movement_type,       # 'straight_right' or 'left_uturn'
                    'approach_orientation': approach_orientation  # 'NS' or 'EW'
                })

    # Rebuild traffic light phases for each intersection
    for tl_logic in root.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        connections = connections_db.get(tl_id, [])

        if not connections:
            continue

        # Group connections by approach direction and movement type
        ns_straight_right = []  # North/South straight+right
        ns_left_uturn = []      # North/South left+u-turn
        ew_straight_right = []  # East/West straight+right
        ew_left_uturn = []      # East/West left+u-turn

        for conn in connections:
            approach_orientation = conn['approach_orientation']  # 'NS' or 'EW'
            movement = conn['movement_type']                     # 'straight_right' or 'left_uturn'
            link_idx = conn['link_index']

            # Group by approach orientation and movement type
            if approach_orientation == 'NS':
                if movement == 'straight_right':
                    ns_straight_right.append(link_idx)
                else:
                    ns_left_uturn.append(link_idx)
            elif approach_orientation == 'EW':
                if movement == 'straight_right':
                    ew_straight_right.append(link_idx)
                else:
                    ew_left_uturn.append(link_idx)

        # Determine state string length (number of connections)
        num_links = max([c['link_index'] for c in connections]) + 1

        # Build phase states
        phases = []

        # Phase 1: N/S straight+right green (30s)
        state1 = build_state_string(num_links, ns_straight_right)
        phases.append({'duration': PARTIAL_OPPOSITES_STRAIGHT_RIGHT_GREEN, 'state': state1})
        phases.append({'duration': PARTIAL_OPPOSITES_YELLOW, 'state': convert_to_yellow(state1)})

        # Phase 2: N/S left+u-turn green (9s)
        state2 = build_state_string(num_links, ns_left_uturn)
        phases.append({'duration': PARTIAL_OPPOSITES_LEFT_UTURN_GREEN, 'state': state2})
        phases.append({'duration': PARTIAL_OPPOSITES_YELLOW, 'state': convert_to_yellow(state2)})

        # Phase 3: E/W straight+right green (30s)
        state3 = build_state_string(num_links, ew_straight_right)
        phases.append({'duration': PARTIAL_OPPOSITES_STRAIGHT_RIGHT_GREEN, 'state': state3})
        phases.append({'duration': PARTIAL_OPPOSITES_YELLOW, 'state': convert_to_yellow(state3)})

        # Phase 4: E/W left+u-turn green (9s)
        state4 = build_state_string(num_links, ew_left_uturn)
        phases.append({'duration': PARTIAL_OPPOSITES_LEFT_UTURN_GREEN, 'state': state4})
        phases.append({'duration': PARTIAL_OPPOSITES_YELLOW, 'state': convert_to_yellow(state4)})

        # Replace phases in traffic light logic
        for phase in tl_logic.findall('phase'):
            tl_logic.remove(phase)

        for phase_data in phases:
            phase_elem = ET.Element('phase', {
                'duration': str(phase_data['duration']),
                'state': phase_data['state']
            })
            tl_logic.append(phase_elem)

    # Write modified traffic light file
    tree.write(CONFIG.network_tll_file, encoding='utf-8', xml_declaration=True)
    print(f"✓ Converted traffic lights to partial_opposites (8 phases per intersection, 90s cycle)")


# Summary of Approach:
#
# Movement Classification: Calculates turn angles from edge geometry
#   - Determines edge directions from junction IDs (N, S, E, W)
#   - Calculates angle between incoming and outgoing directions
#   - Classifies based on angle:
#     * Right turn: ~-90°
#     * Straight: ~0°
#     * Left turn: ~+90°
#     * U-turn: ~180°
#   - Groups: straight+right (-90° to +10°) vs left+u-turn (+10° to +180°)
#
# Approach Orientation Classification: Parses junction IDs from edge names
#   - Format: "A0B0_H_s" → junctions A0 to B0
#   - Determines which way the edge is heading based on junction positions
#   - Groups opposing approaches: N/S vs E/W
#
# Phase Durations: Uses constants for maintainability
#   - Straight+right: 30s green (PARTIAL_OPPOSITES_STRAIGHT_RIGHT_GREEN)
#   - Left+u-turn: 9s green (PARTIAL_OPPOSITES_LEFT_UTURN_GREEN)
#   - Yellow: 3s (PARTIAL_OPPOSITES_YELLOW)
#   - Total cycle: 90 seconds (same as opposites)


def get_edge_direction(edge_id: str) -> str:
    """Determine which cardinal direction an edge is heading

    Parses junction IDs from edge name to determine direction of travel.

    Grid junction naming: {row_letter}{column_number}
    - Rows: A (north) → E (south)
    - Columns: 0 (west) → 4 (east)

    Examples:
        'A0A1_H_s' → 'E' (moving east: column 0→1)
        'A1A0_H_s' → 'W' (moving west: column 1→0)
        'A0B0_H_s' → 'S' (moving south: row A→B)
        'B0A0_H_s' → 'N' (moving north: row B→A)

    Returns:
        'N', 'S', 'E', 'W': Cardinal direction of edge travel
        'unknown': If direction cannot be determined
    """
    import re

    # Remove suffixes
    base_edge = edge_id.split('_')[0]

    # Extract junction IDs
    junctions = re.findall(r'[A-Z]\d+', base_edge)

    if len(junctions) != 2:
        return 'unknown'

    from_junction, to_junction = junctions
    from_row = from_junction[0]
    from_col = int(from_junction[1:])
    to_row = to_junction[0]
    to_col = int(to_junction[1:])

    # Determine direction based on junction change
    if from_row == to_row:  # Same row = horizontal movement
        if to_col > from_col:
            return 'E'  # Moving east (increasing column)
        elif to_col < from_col:
            return 'W'  # Moving west (decreasing column)
    elif from_col == to_col:  # Same column = vertical movement
        if to_row > from_row:
            return 'S'  # Moving south (increasing row letter)
        elif to_row < from_row:
            return 'N'  # Moving north (decreasing row letter)

    return 'unknown'


def get_edge_orientation(edge_id: str) -> str:
    """Determine if edge is vertical (N/S) or horizontal (E/W)

    Used for grouping opposing approaches together.

    Returns:
        'NS': North-South (vertical) edge
        'EW': East-West (horizontal) edge
    """
    direction = get_edge_direction(edge_id)
    if direction in ['N', 'S']:
        return 'NS'
    elif direction in ['E', 'W']:
        return 'EW'
    else:
        return 'EW'  # Default fallback


def calculate_turn_angle(from_dir: str, to_dir: str) -> float:
    """Calculate turn angle from one direction to another

    Args:
        from_dir: Incoming edge direction (N, S, E, W)
        to_dir: Outgoing edge direction (N, S, E, W)

    Returns:
        Angle in degrees (-180 to +180):
        - -90°: Right turn
        - 0°: Straight
        - +90°: Left turn
        - ±180°: U-turn
    """
    direction_angles = {'N': 0, 'E': 90, 'S': 180, 'W': 270}

    if from_dir not in direction_angles or to_dir not in direction_angles:
        return 0  # Default to straight

    from_angle = direction_angles[from_dir]
    to_angle = direction_angles[to_dir]

    # Calculate relative turn angle
    turn = to_angle - from_angle

    # Normalize to -180 to +180 range
    while turn > 180:
        turn -= 360
    while turn < -180:
        turn += 360

    return turn


def classify_movement(turn_angle: float) -> str:
    """Classify movement type for partial_opposites strategy

    Args:
        turn_angle: Turn angle in degrees

    Returns:
        'straight_right': Straight or right turn (-90° to +10°)
        'left_uturn': Left turn or U-turn (+10° to +180°)
    """
    # Threshold at +10° to allow slight angle tolerance for "straight"
    if turn_angle <= 10.0:
        return 'straight_right'
    else:
        return 'left_uturn'


def build_state_string(num_links: int, green_link_indices: list) -> str:
    """Build SUMO traffic light state string

    Args:
        num_links: Total number of connections
        green_link_indices: List of connection indices that get green

    Returns:
        State string like "GGrrGGrr" where:
        - 'G': Green (priority)
        - 'g': Green (no priority)
        - 'r': Red
    """
    state = ['r'] * num_links
    for idx in green_link_indices:
        if 0 <= idx < num_links:
            state[idx] = 'G'
    return ''.join(state)


def convert_to_yellow(green_state: str) -> str:
    """Convert green state to yellow transition state

    Changes 'G' to 'y' (yellow), keeps 'r' as 'r'
    """
    return green_state.replace('G', 'y').replace('g', 'y')
```

**Risk**: High - Core traffic light logic
**Testing**: Must validate with SUMO-GUI that phases work correctly

---

### Step 7: Add Integration Point (`src/network/generate_grid.py`)

Find the integration point after `convert_to_incoming_strategy()` call (around line 384-385) and add:

```python
    # Convert traffic light phasing based on strategy
    if traffic_light_strategy == "incoming":
        convert_to_incoming_strategy()
    elif traffic_light_strategy == "partial_opposites":  # NEW
        convert_to_partial_opposites_strategy()         # NEW
```

**Risk**: Low - Just adding a conditional branch

---

### Step 8: Update Network Validation (`src/validate/validate_network.py`)

#### 8.1. Update Lane Count Validation (lines 203-210)

```python
# OLD:
if num_lanes < 1 or num_lanes > 5:
    errors.append(f"Edge {edge_id} has invalid lane count: {num_lanes}")

# NEW:
from src.constants import MIN_LANES_FOR_TL_STRATEGY
min_required_lanes = MIN_LANES_FOR_TL_STRATEGY.get(traffic_light_strategy, 1)
if num_lanes < min_required_lanes or num_lanes > 5:
    errors.append(f"Edge {edge_id} has invalid lane count: {num_lanes} "
                  f"(min {min_required_lanes} for {traffic_light_strategy} strategy)")
```

#### 8.2. Add Traffic Light Phase Validation (lines 258-295 area)

Add new validation function:

```python
def validate_partial_opposites_phases(traffic_light_strategy: str) -> List[str]:
    """Validate traffic light phases for partial_opposites strategy"""
    errors = []

    if traffic_light_strategy != "partial_opposites":
        return errors  # Only validate for partial_opposites

    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(CONFIG.network_tll_file)
        root = tree.getroot()

        for tl_logic in root.findall('tlLogic'):
            tl_id = tl_logic.get('id')
            phases = tl_logic.findall('phase')

            # partial_opposites should have 8 phases (4 green + 4 yellow)
            if len(phases) != 8:
                errors.append(f"Traffic light {tl_id} has {len(phases)} phases, "
                            f"expected 8 for partial_opposites strategy")

            # Verify phase durations (should be 30, 3, 9, 3, 30, 3, 9, 3)
            # Phase 1: N/S straight+right (30s), Phase 2: yellow (3s)
            # Phase 3: N/S left+u-turn (9s), Phase 4: yellow (3s)
            # Phase 5: E/W straight+right (30s), Phase 6: yellow (3s)
            # Phase 7: E/W left+u-turn (9s), Phase 8: yellow (3s)
            expected_durations = [30, 3, 9, 3, 30, 3, 9, 3]
            for i, phase in enumerate(phases):
                duration = int(phase.get('duration', 0))
                if i < len(expected_durations) and duration != expected_durations[i]:
                    errors.append(f"Traffic light {tl_id} phase {i} has duration {duration}, "
                                f"expected {expected_durations[i]}")

    except Exception as e:
        errors.append(f"Error validating traffic light phases: {str(e)}")

    return errors
```

Call this validation in the main validation flow.

**Risk**: Medium - Adds new validation logic

---

## Testing Protocol

### Test 1: Basic 3x3 Grid Test

**Command**:
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --end-time 1800 \
  --traffic_light_strategy partial_opposites \
  --lane_count realistic \
  --gui
```

**Expected Results**:
- ✓ Network generates without errors
- ✓ All edges have ≥2 lanes
- ✓ `.tll.xml` contains 8 phases per traffic light
- ✓ SUMO-GUI loads and runs simulation
- ✓ Vehicles flow correctly through intersections
- ✓ No TraCI errors or warnings

**Manual Verification in SUMO-GUI**:
1. Right-click on traffic light → "Show Phases"
2. Verify 8 phases displayed
3. Verify phase durations: 30s, 3s, 9s, 3s, 30s, 3s, 9s, 3s (total = 90s cycle)
4. Observe that N/S straight+right go together in phase 1 (30s green)
5. Observe that N/S left turns go in phase 3 (9s green)
6. Observe that E/W straight+right go together in phase 5 (30s green)
7. Observe that E/W left turns go in phase 7 (9s green)

---

### Test 2: Regression Test - Opposites Strategy (CRITICAL)

**Command**:
```bash
# Before implementation: Save output
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --traffic_light_strategy opposites \
  --lane_count realistic \
  --seed 42

# After implementation: Compare output
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --traffic_light_strategy opposites \
  --lane_count realistic \
  --seed 42
```

**Expected Results**:
- ✓ `.tll.xml` file is IDENTICAL before/after implementation
- ✓ Lane counts are IDENTICAL before/after implementation
- ✓ All network files (`.net.xml`, `.edg.xml`, etc.) are IDENTICAL

**Verification Method**:
```bash
# Generate checksums before implementation
md5sum workspace_before/grid.tll.xml
md5sum workspace_before/grid.net.xml

# Generate checksums after implementation
md5sum workspace_after/grid.tll.xml
md5sum workspace_after/grid.net.xml

# Should be IDENTICAL
```

**CRITICAL**: If this test fails, the implementation has unintended side effects on existing strategies.

---

### Test 3: Regression Test - Incoming Strategy (CRITICAL)

Same as Test 2, but with `--traffic_light_strategy incoming`

**Expected Results**:
- ✓ IDENTICAL output before/after implementation

---

### Test 4: Edge Case - Fixed 1 Lane (Should ERROR)

**Command**:
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --traffic_light_strategy partial_opposites \
  --lane_count "fixed 1"
```

**Expected Result**:
- ✗ Clear error message:
  ```
  ValidationError: partial_opposites strategy requires minimum 2 lanes per edge.
  You specified: fixed 1.
  Use '--lane_count fixed 2' or higher, or use 'realistic'/'random' algorithms.
  ```
- ✗ Pipeline stops before network generation

---

### Test 5: Edge Case - Fixed 2 Lanes (Should Work)

**Command**:
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --end-time 1800 \
  --traffic_light_strategy partial_opposites \
  --lane_count "fixed 2" \
  --gui
```

**Expected Results**:
- ✓ Network generates successfully
- ✓ All edges have exactly 2 lanes
- ✓ Simulation runs correctly

---

### Test 6: Random Lane Count

**Command**:
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --end-time 1800 \
  --traffic_light_strategy partial_opposites \
  --lane_count random \
  --gui
```

**Expected Results**:
- ✓ Network generates successfully
- ✓ All edges have 2 or 3 lanes (randomly distributed)
- ✓ No edges with 1 lane

**Verification**:
```bash
# Check lane counts in network file
grep 'numLanes' workspace/grid.edg.xml
# All should be numLanes="2" or numLanes="3"
```

---

### Test 7: Traffic Control Method Compatibility

Test with all traffic control methods:

```bash
# Tree Method
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --end-time 1800 \
  --traffic_light_strategy partial_opposites \
  --traffic_control tree_method \
  --gui

# Actuated
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --end-time 1800 \
  --traffic_light_strategy partial_opposites \
  --traffic_control actuated \
  --gui

# Fixed
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 3 \
  --num_vehicles 100 \
  --end-time 1800 \
  --traffic_light_strategy partial_opposites \
  --traffic_control fixed \
  --gui
```

**Expected Results**:
- ✓ All three methods work without errors
- ✓ Tree Method: Uses phases from `.tll.xml` as basis for optimization
- ✓ Actuated: SUMO manages phases dynamically
- ✓ Fixed: Uses exact phase durations from `.tll.xml`

---

### Test 8: Full Integration Test (5x5 Grid)

**Command**:
```bash
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 150 \
  --num_vehicles 800 \
  --end-time 3600 \
  --traffic_light_strategy partial_opposites \
  --traffic_control tree_method \
  --departure_pattern six_periods \
  --routing_strategy 'shortest 70 realtime 30' \
  --gui
```

**Expected Results**:
- ✓ Larger network generates successfully
- ✓ All 25 intersections have 8-phase traffic lights
- ✓ Simulation completes without errors
- ✓ Performance metrics are reasonable

**Performance Checks**:
- Average travel time should be reasonable (< 500s)
- Completion rate should be high (> 90%)
- No deadlocks or infinite queues

---

## Known Limitations

### 1. Cycle Time and Phase Duration Trade-offs

- **Cycle time**: 90 seconds (same as `opposites` strategy)
- **Phase durations**:
  - Straight+right: 30s green per direction (same total as opposites)
  - Left+u-turn: 9s green per direction (significantly less)
- **Green time allocation**:
  - `opposites`: 42s per direction (all movements together) = 47% of cycle
  - `partial_opposites`: 30s + 9s = 39s per direction (split by movement) = 43% of cycle
- **Impact**:
  - Similar overall throughput to `opposites` for straight/right movements
  - Reduced capacity for left turns (may cause queuing if left-turn demand is high)
  - Better than `incoming` strategy which gives only 30s total per direction
- **Benefit**: Eliminates left-turn vs straight-traffic conflicts without dramatically increasing cycle time

### 2. Minimum Lane Requirement

- **Requires**: 2+ lanes per edge
- **Impact**: Not suitable for very low-traffic scenarios where 1 lane is sufficient
- **Recommendation**: Use `opposites` for single-lane networks

### 3. Grid Network Assumptions

- **Direction classification** assumes orthogonal grid layout with junction naming convention
- Uses junction IDs to determine edge direction and orientation
- Junction format: `{row_letter}{column_number}` (e.g., A0, B2, C3)
  - Rows: A (north) → E (south)
  - Columns: 0 (west) → 4 (east)
- **Movement classification** uses geometric turn angle calculations
  - Calculates angles between incoming and outgoing edge directions
  - Assumes standard cardinal directions (N, S, E, W) and orthogonal intersections
  - Straight+right: -90° to +10° turn angles
  - Left+u-turn: +10° to +180° turn angles
- **Limitation**: Only works with grid networks using the standard junction naming convention
- **Future work**: Generalize for non-grid topologies and custom junction naming

### 4. U-Turn Handling

- U-turns grouped with left turns (same phase)
- If network has high U-turn demand, this may create congestion in left-turn phase
- **Mitigation**: Consider restricting U-turns in network generation if needed

---

## Backward Compatibility Statement

**CRITICAL**: This implementation has **ZERO impact** on existing traffic light strategies.

### Guarantees:

1. **`opposites` strategy**:
   - Lane count behavior unchanged (1-3 lanes as before)
   - Traffic light phases unchanged (2-phase system)
   - All existing simulations work identically

2. **`incoming` strategy**:
   - Lane count behavior unchanged (1-3 lanes as before)
   - Traffic light phases unchanged (1 phase per incoming edge)
   - All existing simulations work identically

3. **Default behavior**:
   - Default strategy is now `partial_opposites` (changed from `opposites`)
   - Users can explicitly request `opposites` or `incoming` if needed
   - No automatic conversions or surprises

### Testing Verification:

The regression tests (Test 2 and Test 3) MUST pass with identical outputs before/after implementation. If any differences are detected, the implementation has introduced unintended side effects and must be fixed.

---

## Files to Modify Summary

| # | File | Type | Lines | Risk | Phase |
|---|------|------|-------|------|-------|
| 1 | `docs/PARTIAL_OPPOSITES_STRATEGY.md` | NEW | ~600 | N/A | 1 |
| 2 | `src/constants.py` | Add | ~15 | Low | 1 |
| 3 | `src/args/parser.py` | Modify | ~3 | Low | 1 |
| 4 | `src/validate/validate_arguments.py` | Add | ~25 | Low | 1 |
| 5 | `src/network/lane_counts.py` | Modify | ~20 | Medium | 2 |
| 6 | `src/network/split_edges_with_lanes.py` | Modify | ~5 | Low | 2 |
| 7 | `src/network/generate_grid.py` | Add | ~200 | High | 3 |
| 8 | `src/validate/validate_network.py` | Modify | ~30 | Medium | 4 |
| 9 | `CLAUDE.md` | Modify | ~15 | Low | 6 |

**Total**: ~310 new/modified lines across 9 files (excluding this documentation)

---

## Implementation Timeline

**Estimated Total**: 10-12 hours of focused development time

- **Phase 1** (Foundation): 1 hour
- **Phase 2** (Lane Enforcement): 2 hours
- **Phase 3** (Traffic Light Logic): 3-4 hours
- **Phase 4** (Validation): 1 hour
- **Phase 5** (Testing): 2-3 hours
- **Phase 6** (Documentation): 30 minutes

---

## Success Criteria

Implementation is considered complete when:

1. ✓ All 8 test cases pass
2. ✓ Regression tests confirm `opposites` and `incoming` are unchanged
3. ✓ SUMO-GUI shows correct 8-phase traffic lights
4. ✓ All edges have ≥2 lanes when using `partial_opposites`
5. ✓ Clear error message for incompatible lane configurations
6. ✓ Works with all traffic control methods (tree_method, actuated, fixed, rl)
7. ✓ Documentation is complete and accurate
8. ✓ `CLAUDE.md` updated with examples

---

## Rollback Plan

If critical issues are discovered after implementation:

1. **Quick Disable**: Remove `"partial_opposites"` from CLI choices in `src/args/parser.py`
2. **Preserve Code**: Keep conversion function for future debugging
3. **Revert Lane Logic**: Restore original `lane_counts.py` if side effects detected
4. **Document Issues**: Update this file with discovered problems
5. **Future Fix**: Address issues in separate iteration

---

## Future Enhancements

After initial implementation is stable:

1. **Adaptive Phase Durations**: Adjust green times based on traffic demand
2. **Non-Orthogonal Networks**: Generalize angle calculation for irregular networks
3. **Protected Left Turns**: Add dedicated left-turn lanes (3+ lane requirement)
4. **Performance Analysis**: Compare throughput vs `opposites` and `incoming` strategies
5. **GUI Integration**: Add visual explanation of phase system in web GUI

---

*End of Implementation Roadmap*
