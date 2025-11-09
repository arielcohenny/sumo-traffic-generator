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
