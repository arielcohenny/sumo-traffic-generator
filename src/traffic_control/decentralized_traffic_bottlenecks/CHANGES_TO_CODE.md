# Changes to Tree Method Code

This document tracks modifications made to the original Tree Method (decentralized-traffic-bottlenecks) implementation.

## 2025-11-15: Fix Division by Zero in Head Cost Calculation

**File**: `shared/classes/link.py`
**Method**: `calc_heads_costs()`
**Line**: ~138-149

### Problem
The algorithm crashed with `ZeroDivisionError` when calculating head costs in scenarios where no vehicles were present on any head edges. This occurred in:
- Large networks with removed junctions
- Low-density traffic areas
- Peripheral edges with minimal traffic
- Early/late simulation periods

**Error**:
```python
heads_costs[head_name] = all_heads[head_name].iterations_vehicle_count[-1] / \
    heads_vehicles_count_sum * link_cost
# ZeroDivisionError when heads_vehicles_count_sum = 0
```

### Solution
Added zero-division protection that distributes costs equally among head edges when no vehicle data is available:

```python
if heads_vehicles_count_sum == 0:
    # No vehicles - distribute cost equally among heads
    equal_cost = link_cost / len(self.head_names) if self.head_names else 0
    for head_name in self.head_names:
        heads_costs[head_name] = equal_cost
else:
    # Normal case - distribute proportionally to vehicle counts
    for head_name in self.head_names:
        heads_costs[head_name] = all_heads[head_name].iterations_vehicle_count[-1] / \
            heads_vehicles_count_sum * link_cost
```

### Rationale
- **Equal distribution** is the fairest assumption when no vehicle data exists
- Prevents crashes while maintaining algorithm correctness
- Minimal impact: only affects edge cases with zero vehicles
- Normal operation remains unchanged when vehicle data is available

### Testing
Tested with large network (6x6 grid, 17000 vehicles, 7200s simulation) with junction removal. Simulation completes successfully without division by zero errors.

### Changed By
Ariel - November 15, 2025

---

## Change to net_data_builder.py

Change to net_data_builder.py was a critical bug fix in the
print_junction() method to handle empty heads arrays properly in JSON
generation.

### The Fix (lines 196-200):

Fix: Only remove trailing comma if heads exist, otherwise close empty array

if len(phase.heads) > 0:
junction_string = junction_string[:-2] + ']},'
else:
junction_string = junction_string + ']},'

### What the bug was:

- Before: When a traffic light phase had no heads (empty array), the code would
  try to remove the last 2 characters [:-2] from "heads": [, resulting in
  malformed JSON like "heads":]
- After: The fix checks if heads exist before removing the trailing comma,
  properly generating "heads": [] for empty arrays

### Why this was critical:

This bug was causing the "Expecting value: line 170 column 102 (char 24192)"
JSON parsing error that prevented Tree Method from working with our synthetic
grid networks. The malformed JSON would break the Tree Method's network data
loading process.

### Context:

This fix was part of commit bc84a5d "Fixing benchmarks from the tree method
experiments" which reduced the file size by 65 lines (87 → 22
insertions/deletions), indicating significant cleanup along with this critical
bug fix.

## Change to graph.py

Added lists to prevent double counting that rarely happens when many vehicles are teleported (and sometimes brought back):
ended_ids_list
started_ids_list

## Change to controller.py - Tree Method Decision Logging

Added console logging to display Tree Method traffic light phase duration decisions in real-time during simulation.

### The Change (lines 152-166):

Added logging after `_populate_shared_phase_durations()` to display:
- Iteration header with step number, timestamp (HH:MM:SS), and number of junctions being calculated
- All phase durations for each junction in compact array format

### Example Output:

```
🔧 step=120 | time=00:02:00 | Tree Method calculating phases for 16 junctions
🚦 step=120 | J1: [25s, 30s, 20s, 15s]
🚦 step=120 | J2: [18s, 22s, 28s, 12s]
🚦 step=120 | J3: [22s, 28s, 18s, 12s]
...
```

### Purpose:

- **Real-time visibility**: Allows users to see Tree Method decisions as they happen during simulation
- **Debugging**: Makes it easier to understand how Tree Method adapts phase durations based on traffic conditions
- **Research**: Provides human-readable output complementing the detailed CSV logger
- **Verbosity**: Displays all phase durations (typically 4 phases) for each junction at every decision interval (default 60 seconds)

### Technical Details:

- Uses Python's `logging.info()` for console output (standard logging level)
- Timestamps converted from simulation steps to HH:MM:SS format for readability
- Junction IDs sorted alphabetically for consistent output ordering
- Durations displayed in compact array format: [phase0_duration, phase1_duration, ...]
- Non-functional change: Only adds logging, does not modify algorithm behavior
- Complements existing `BottleneckCSVLogger` which captures detailed bottleneck data to CSV file

### Context:

This enhancement was requested to improve visibility into Tree Method's decision-making process. The existing CSV logger already captures all this data and more (bottleneck scores, vehicle counts, etc.), but the console output provides immediate feedback during simulation runs and is easier to monitor in real-time.
