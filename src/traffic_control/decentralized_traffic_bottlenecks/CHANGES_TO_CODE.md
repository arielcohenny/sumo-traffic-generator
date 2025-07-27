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
