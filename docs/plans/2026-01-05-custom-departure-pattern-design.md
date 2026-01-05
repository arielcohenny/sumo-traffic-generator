# Custom Departure Pattern Design

**Date:** 2026-01-05
**Status:** Approved

## Overview

Replace the existing `rush_hours` departure pattern with a more flexible `custom` pattern that allows precise control over when vehicles depart using absolute clock times and percentage allocations.

## Syntax

```
--departure_pattern "custom:HH:MM-HH:MM,percent;HH:MM-HH:MM,percent"
```

### Examples

```bash
# 40% at 9:00-9:30, 30% at 10:00-10:45, remaining 30% fills gaps
--departure_pattern "custom:9:00-9:30,40;10:00-10:45,30"

# Single rush hour: 70% at morning rush, 30% distributed elsewhere
--departure_pattern "custom:8:00-9:30,70"

# Multiple peaks
--departure_pattern "custom:7:00-8:00,25;12:00-13:00,25;17:00-18:00,25"
```

### Key Behaviors

- Times are absolute clock times (9:00 = 9:00 AM)
- Must fall within simulation window [start_time, start_time + duration]
- Trailing semicolon allowed but not required
- Percentages need not sum to 100%; remainder auto-distributed to gaps

## Validation Rules

### Percentage Validation
- Sum of specified percentages must be <= 100%
- Error if exceeded: `"Specified percentages sum to 110%, must be <= 100%"`

### Time Range Validation
- Window start must be before window end
- Error: `"Invalid window 9:30-9:00: start time must be before end time"`

### Simulation Bounds Validation
- All windows must fall within [start_time, start_time + duration]
- Given `--start_time_hour 8.0 --end-time 18000` (5 hours, ends 13:00):
  - Valid: `9:00-9:30`
  - Invalid: `7:00-8:00` (before start)
  - Invalid: `12:00-14:00` (extends past end)
- Error: `"Window 7:00-8:00 is outside simulation range 8:00-13:00"`

### Overlap Validation
- Windows must not overlap
- Error: `"Windows 9:00-10:00 and 9:30-10:30 overlap"`

### Format Validation
- Times must be valid HH:MM format (00:00 to 23:59)
- Percentages must be positive integers
- Error: `"Invalid time format '9:75': minutes must be 0-59"`

## Distribution Algorithm

### Step 1: Parse and Validate Windows

```
Input: "custom:9:00-9:30,40;10:00-10:45,30"
Parsed: [(9:00, 9:30, 40%), (10:00, 10:45, 30%)]
```

### Step 2: Calculate Rest Percentage and Windows

```
Specified: 40% + 30% = 70%
Rest: 100% - 70% = 30%

Simulation range: 8:00 - 13:00
Rest windows (gaps between specified + edges):
  - 8:00-9:00   (60 min)
  - 9:30-10:00  (30 min)
  - 10:45-13:00 (135 min)
Total rest duration: 225 min
```

### Step 3: Allocate Vehicles to Windows

```
Total vehicles: 500

Specified windows:
  - 9:00-9:30:   500 * 40% = 200 vehicles
  - 10:00-10:45: 500 * 30% = 150 vehicles

Rest windows (30% = 150 vehicles, proportional by duration):
  - 8:00-9:00:   150 * (60/225)  = 40 vehicles
  - 9:30-10:00:  150 * (30/225)  = 20 vehicles
  - 10:45-13:00: 150 * (135/225) = 90 vehicles
```

### Step 4: Generate Departure Times

- Uniform distribution within each window
- Sorted chronologically in final output

## Scope & Interactions

### What `--departure_pattern custom:...` Controls
- **WHEN** vehicles depart (temporal distribution)
- Nothing else

### What Other Arguments Control
- **WHERE** vehicles depart from: `--attractiveness` (land_use, poisson, iac)
- **WHERE** vehicles go to: `--attractiveness` + route patterns
- **HOW** vehicles route: `--routing_strategy` (shortest, realtime, fastest)
- **WHAT TYPE** of vehicles: `--vehicle_types` (passenger, public)
- **HOW MANY** vehicles: `--num_vehicles`

### Interaction with Existing Systems
- Departure times feed into `determine_attractiveness_phase_from_departure_time()` for edge selection
- The 4-phase temporal system (morning_peak, midday_offpeak, evening_peak, night_low) still applies for attractiveness weighting
- Route pattern bias (`assign_route_pattern_with_temporal_bias()`) uses departure time
- Custom windows work with both private and public traffic seeds

### Vehicle Types
- Same pattern applies to both private (passenger) and public vehicles
- Consistent with existing behavior of other patterns

## Implementation Changes

### Files to Modify

1. **`src/constants.py`**
   - Add `DEPARTURE_PATTERN_CUSTOM = "custom"`
   - Remove or deprecate `RUSH_HOURS_PREFIX`

2. **`src/traffic/builder.py`**
   - Replace `_calculate_rush_hours_deterministic()` with `_calculate_custom_deterministic()`
   - Update `calculate_temporal_departure_times()` to handle `custom:` pattern
   - Add parsing function `_parse_custom_pattern(pattern: str) -> List[TimeWindow]`
   - Add rest window computation

3. **`src/validate/validate_arguments.py`**
   - Add validation for `custom:` pattern syntax
   - Validate windows against simulation bounds
   - Check for overlaps and percentage sum

4. **`src/args/parser.py`**
   - Update help text for `--departure_pattern`

5. **`docs/specification/command-line-interface.md`**
   - Update documentation for new syntax

6. **`src/ui/parameter_widgets.py`** (if GUI supports it)
   - Update dropdown/input for departure pattern
