# Custom Edge Lane Definition System - Complete Implementation Plan

## Overview
Add capability for users to manually specify lane configurations for specific edges in synthetic grids, overriding the automatic lane assignment algorithms. This is a post-processing step that occurs after the current `split_edges_with_flow_based_lanes()` completes.

## Configuration Syntax
```bash
# CLI argument format
--custom_lanes "A1B1=tail:2,B1B0:1,B1C1:2;A2B2=tail:3,B2C2:2,B2B1:1"

# Alternative file-based approach for complex scenarios  
--custom_lanes_file custom_lanes.txt
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

**Example**: User specifies `A1B1=tail:2,B1B0:1,B1C1:2`
```xml
<!-- BEFORE custom lanes -->
<edge id="A1B1" from="A1" to="A1B1_H_node" numLanes="1" ... />
<edge id="A1B1_H_s" from="A1B1_H_node" to="B1" numLanes="4" ... />

<!-- AFTER custom lanes -->  
<edge id="A1B1" from="A1" to="A1B1_H_node" numLanes="2" ... />          <!-- Updated tail -->
<edge id="A1B1_H_s" from="A1B1_H_node" to="B1" numLanes="3" ... />      <!-- Updated head: 1+2=3 lanes -->
```

### 2. `con.xml` Changes
**Strategy**: Delete ALL existing connections for the edge, then recreate with new lane assignments

**DELETE ALL CONNECTIONS**:
```xml
<!-- Remove all existing connections FROM the tail segment -->
<connection from="A1B1" to="A1B1_H_s" fromLane="..." toLane="..." />

<!-- Remove all existing connections FROM the head segment -->  
<connection from="A1B1_H_s" to="B1B0" fromLane="..." toLane="..." />
<connection from="A1B1_H_s" to="B1C1" fromLane="..." toLane="..." />
<connection from="A1B1_H_s" to="B1B2" fromLane="..." toLane="..." />
<connection from="A1B1_H_s" to="B1A1" fromLane="..." toLane="..." />
```

**RECREATE WITH NEW LANE ASSIGNMENTS**:
```xml
<!-- New internal tail→head connections (2 tail lanes → 3 head lanes) -->
<connection from="A1B1" to="A1B1_H_s" fromLane="0" toLane="0"/>
<connection from="A1B1" to="A1B1_H_s" fromLane="1" toLane="1"/>
<connection from="A1B1" to="A1B1_H_s" fromLane="1" toLane="2"/>

<!-- New head→downstream connections (custom specified) -->
<connection from="A1B1_H_s" to="B1B0" fromLane="0" toLane="0"/>         <!-- 1 lane as specified -->
<connection from="A1B1_H_s" to="B1C1" fromLane="1" toLane="0"/>         <!-- 2 lanes as specified -->
<connection from="A1B1_H_s" to="B1C1" fromLane="2" toLane="1"/>
<!-- Other movements keep existing assignments but with updated fromLane -->
<connection from="A1B1_H_s" to="B1B2" fromLane="?" toLane="0"/>         <!-- Need to calculate -->
<connection from="A1B1_H_s" to="B1A1" fromLane="?" toLane="0"/>         <!-- Need to calculate -->
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

<!-- AFTER: B1 might have 18 connections, need state="GGgggrrrrGGgggrrrr" (18 chars) -->
<tlLogic id="B1" type="static" programID="0" offset="0">
    <phase duration="42" state="GGgggrrrrGGgggrrrr" />     <!-- RECALCULATED! -->
    <phase duration="3" state="yyyyyrrrryyyyyrrrr" />      <!-- RECALCULATED! -->
    <phase duration="42" state="rrrrrGGggrrrrrGGgg" />     <!-- RECALCULATED! -->
    <phase duration="3" state="rrrrryyyyrrrrryyy" />       <!-- RECALCULATED! -->
</tlLogic>
```

#### 4b. Traffic Light Connection linkIndex Recalculation (Bottom of file)
**CRITICAL**: ALL connections going into affected junction need linkIndex recalculated!

```xml
<!-- DELETE all traffic light connections for affected junction -->
<connection from="A1B1_H_s" to="B1B0" fromLane="0" toLane="0" tl="B1" linkIndex="0" />
<connection from="A1B1_H_s" to="B1C1" fromLane="1" toLane="0" tl="B1" linkIndex="1" />
<connection from="A1B1_H_s" to="B1B2" fromLane="2" toLane="0" tl="B1" linkIndex="2" />
<connection from="A1B1_H_s" to="B1A1" fromLane="3" toLane="0" tl="B1" linkIndex="3" />
<!-- AND ALL OTHER CONNECTIONS TO B1 FROM OTHER EDGES -->
<connection from="B0B1_H_s" to="B1C1" fromLane="0" toLane="0" tl="B1" linkIndex="4" />
<!-- ... dozens more connections ... -->

<!-- RECREATE with recalculated linkIndex values -->
<connection from="A1B1_H_s" to="B1B0" fromLane="0" toLane="0" tl="B1" linkIndex="0" />
<connection from="A1B1_H_s" to="B1C1" fromLane="1" toLane="0" tl="B1" linkIndex="1" />
<connection from="A1B1_H_s" to="B1C1" fromLane="2" toLane="0" tl="B1" linkIndex="2" />    <!-- NEW -->
<!-- ALL other connections need linkIndex shifted accordingly -->
<connection from="B0B1_H_s" to="B1C1" fromLane="0" toLane="0" tl="B1" linkIndex="5" />    <!-- Was 4, now 5 -->
```

## Junction-Wide Impact Analysis

**CRITICAL INSIGHT**: Changing lanes on ONE edge affects the ENTIRE junction's traffic light system!

When we modify `A1B1`, we must recalculate:
1. **All connections INTO junction B1**:
   - `A1B1_H_s` → B1 (modified)
   - `B0B1_H_s` → B1 (linkIndex shifted)  
   - `B2B1_H_s` → B1 (linkIndex shifted)
   - `C1B1_H_s` → B1 (linkIndex shifted)

2. **Traffic light state string for B1**: Must be completely regenerated based on new total connection count

3. **All linkIndex values for B1**: Must be reassigned sequentially

## Implementation Architecture

### File Structure
```
src/network/custom_lanes.py          # Main implementation
src/config.py                        # Add CustomLaneConfig class
src/args/parser.py                   # Add CLI arguments
src/validate/validate_arguments.py   # Add validation
src/pipeline/standard_pipeline.py    # Integration point
```

### New Function: `apply_custom_lane_configs()`
**Location**: `src/network/custom_lanes.py`
**Called from**: `StandardPipeline` after step 3 (edge splitting)

### Detailed Algorithm

#### Phase 1: Parse and Validate
1. Parse custom lane specification from CLI argument
2. Validate edge existence in current XML files
3. Validate movement specifications match available movements
4. Validate lane counts are within bounds (1-3)

#### Phase 2: Calculate Impact
1. For each custom edge, identify affected junctions (downstream)
2. Calculate new head lane requirements based on movement specifications
3. Determine which junctions need traffic light recalculation

#### Phase 3: Update XML Files
1. **Update `edg.xml`**: Modify `numLanes` for tail and head segments
2. **Update `con.xml`**: 
   - Delete all connections for affected edges
   - Regenerate internal tail→head connections
   - Regenerate head→downstream connections with custom lane assignments
   - Preserve other movements with updated lane assignments
3. **Update `tll.xml`**:
   - For each affected junction:
     - Count total connections going into junction
     - Regenerate state strings for all phases
     - Delete all traffic light connections for junction
     - Recreate all traffic light connections with sequential linkIndex

#### Phase 4: Spatial Logic Preservation
1. Maintain existing spatial movement logic (right turns → rightmost lanes, etc.)
2. Apply custom lane counts within spatial constraints
3. Ensure movement distribution follows existing algorithms

## File-by-File Implementation Details

### 1. `src/args/parser.py`
**Location**: Add to `_add_network_arguments()` function around line 74

```python
parser.add_argument(
    "--custom_lanes",
    type=str,
    help="Custom lane definitions for specific edges (format: 'EdgeID=tail:N,ToEdge1:N,ToEdge2:N;EdgeID2=...')"
)
```

### 2. `src/validate/validate_arguments.py`
**Location**: Add validation function and call it in `validate_arguments()` around line 44

```python
def _validate_custom_lanes(custom_lanes: str) -> None:
    """Validate custom lanes argument format and values."""
    if not custom_lanes:
        return
    
    # Parse and validate format: EdgeID=tail:N,ToEdge:N;EdgeID2=...
    # Validate edge ID formats (A1B1 pattern)
    # Validate lane counts (1-3 range)
    # Validate movement specifications
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
**Main implementation file**

```python
def apply_custom_lane_configs(custom_lane_config: CustomLaneConfig) -> None:
    """Apply custom lane configurations to split edges in XML files."""
    
    # Phase 1: Load and analyze current XML files
    # Phase 2: Calculate impact and new requirements
    # Phase 3: Update all XML files with new lane configurations
    # Phase 4: Regenerate traffic light logic and connections

def _update_edges_file(edge_configs: Dict) -> None:
    """Update numLanes in edg.xml for custom edges."""

def _update_connections_file(edge_configs: Dict) -> None:
    """Regenerate connections in con.xml for custom edges."""

def _update_traffic_lights_file(affected_junctions: Set[str]) -> None:
    """Regenerate traffic light logic and connections in tll.xml."""

def _calculate_junction_connections(junction_id: str) -> List[Dict]:
    """Calculate all connections going into a specific junction."""

def _generate_traffic_light_state_string(connections: List[Dict]) -> str:
    """Generate traffic light state string based on connection count."""
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

## Expected Complexity and Challenges

### High Complexity Areas
1. **Traffic Light State Generation**: Must understand SUMO's opposites vs incoming phasing logic
2. **Junction-Wide Impact**: One edge change affects entire junction's numbering
3. **Spatial Logic Preservation**: Custom lanes must respect turn angle assignments
4. **linkIndex Recalculation**: Sequential reassignment across all junction connections

### Medium Complexity Areas  
1. **Connection Regeneration**: Following existing tail→head distribution patterns
2. **Movement-to-Lane Assignment**: Applying custom specs within spatial constraints
3. **XML File Coordination**: Ensuring consistency across edg/con/tll files

### Low Complexity Areas
1. **CLI Argument Parsing**: Standard argparse implementation
2. **Configuration Management**: Dataclass-based structure
3. **Validation**: Format and range checking

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

```bash
# Basic custom lane specification
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2,B1B0:1,B1C1:2" \
  --gui

# Multiple edge specifications
env PYTHONUNBUFFERED=1 python -m src.cli \
  --grid_dimension 5 \
  --block_size_m 200 \
  --num_vehicles 800 \
  --custom_lanes "A1B1=tail:2,B1B0:1,B1C1:2;B2C2=tail:3,C2C1:2,C2D2:1" \
  --gui
```

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