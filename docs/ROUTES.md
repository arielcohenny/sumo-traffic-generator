# Vehicle Route Generation System - Implementation Roadmap

## Overview

### Route Types for All Vehicles

All vehicle types (passenger, public) follow four fundamental route patterns:

**In-bound Routes (in)**: Start at boundary edge (attractivness is irrelevant), end at inner edge (phase attractiveness is relevant)
**Out-bound Routes (out)**: Start at inner edge (phase attractiveness is relevant), end at boundry edge (attractivness is irrelevant)
**Inner Routes (inner)**: Start and end at inner edges (phase attractiveness is relevant for both start/end edges)
**Pass-through Routes (pass)**: Start and end at boundry edges (phase attractiveness is irrelevant for both start/end edges)

### Integration with Existing Systems

**Departure Pattern Integration**: Route selection timing follows existing temporal patterns (six_periods, uniform, custom). Morning rush hours favor in-bound routes to business zones, evening rush hours favor out-bound routes from residential areas.

**Attractiveness Method Integration**: Route endpoints are selected using existing attractiveness methods (land_use, poisson, iac). In-bound routes target high-arrival attractiveness inner edges, out-bound routes originate from high-departure attractiveness inner edges.

### Public Transit Specifics

**Fixed Routes**: Public vehicles operate on fixed route definitions that multiple vehicles share over time.
**Temporal Dispatch**: Public vehicles are dispatched on their assigned routes based on departure patterns, with time gaps between vehicles on the same route.

## Configuration Requirements

### Route Pattern Percentages

Each vehicle type requires four percentage arguments to specify route pattern distribution:

- `--passenger-routes "in X out Y inner Z pass W"` where X+Y+Z+W = 100
- `--public-routes "in X out Y inner Z pass W"` where X+Y+Z+W = 100

## Implementation Roadmap:

### Step 1: New Command Line Arguments:

**Arguments to add**:

- `--passenger-routes "in X out Y inner Z pass W"`
- `--public-routes "in X out Y inner Z pass W"`

constants:
DEFAULT_PASSENGER_ROUTES = "in 30 out 30 inner 25 pass 15"
DEFAULT_PUBLIC_ROUTES = "in 25 out 25 inner 35 pass 15"

**Validation of arguments**:

-- Add to validate_arguments(args) a new call to new function \_validate_route_args(args) that validates:
For --passenger-routes/--public-routes: X + Y + Z + W = 100

**Addition to GUI**:

-- Add a new section after the Traffic Generation section: 🚗 Route Pattern Configuration
-- Iff Vehicle Types: Passenger % > 0 Allow to set --passenger-routes
-- Iff Vehicle Types: Public % > 0 Allow to set --public-routes
-- For each option validate that X + Y + Z + W = 100
-- Make sure to add relevant argument setting to the Command Line.

### Step 2: Classify edges

-- In src/network/generate_grid.py.
-- New function classify_edges:
-- we only examine the tail part of edges, namely edges without suffix \_H_s or \_H_node (example: include edge )
-- Should classify edges to inner/boundary
-- Boundary edges: AiAj for all i/j, X0Y0 for all X/Y, XdimYdim for all X/Y where dim is (dimention - 1), MiMj for all i/j where M's val minus A's val is the dimention. Inner edges are all the rest.
-- For example in 5 x 5 grid we have junctions: A0,A1,..A4,B0,...B4,..,E0,..,E4. Boundry edges are: A0A1, A1A0,..., A3A4,A4A3 (AiAj for all i/j) A0B0,B0A0,...,D0E0,E0D0, (X0Y0 for all X/Y), E0E1, E1E0,..., E3E4,E4E3 (XdimYdim for all X/Y where dim is (dimention - 1)), A4B4,B4A4,..,D4E4,E4D4 (MiMj for all i/j where M's val minus A's val is the dimention)

### Step 3: Generate Routes

-- Rewrite generate_vehicle_routes:

- ALL arguments to the function should not have default values. I don't like using default values!!!

#### Implementing passenger routes:

number of routes from each type:
num_in_routes = passenger-routes.in/100 * num*passenger_vehicles
num_out_routes = passenger-routes.out/100 * num*passenger_vehicles
num_inner_routes = passenger-routes.inner/100 * num*passenger_vehicles
num_pass_routes = passenger-routes.pass/100 \* num_passenger_vehicles

Implementation should be similar to as it is now, with the addition that when picking start/end edge consider not only the attractiveness value but also the addition of restricting the start/end edges to pick from:
for in route:
-- strat: only from Boundary edges
-- end: no new restrictions (could be any edge)
for out route:
-- strat: no new restrictions (could be any edge)
-- end: only to boundry edges
for inner route:
-- strat: no new restrictions (could be any edge)
-- end: no new restrictions (could be any edge)
for pass route:
-- strat: only from boundry edges
-- end: only to boundry edges

Don't forget that vehicles also have Routing Strategies like previously: shortest, realtime, etc. But the Routing Strategies is Orthogonal to the Route Patterns System. The Route Patterns Controls WHERE vehicles start/end. The Routing Strategies Control HOW vehicles navigate between those points. Thus after edge selection, the existing routing strategies remain exactly the same. BUT! public routes should always use the shortest path!!!!

For each vehicle, based on the departure time we can use the attractibness values to decide of the start/end point (attractivness has 4 time phases and based on the vehicle's departure time we'll know which one to use). Following is the function that calculates the departure times, it should of course be used before generating the routes.

```python
Helper functions to set departure times (it just pseudo code that explain in general what to do):
//works both for passenger and public vehicles.

      calculate_temporal_departure_times(num_vehicles, departure_pattern, start_time, end_time):
        departure_times = []
        if departure_pattern == uniform:
          interval = (end_time - start_time) / num_vehicles
          for (i = 0, i < num_vehicles, i++):
            departure_times.append(start_time + i\*interval)
        return departure_times

        if departure_pattern == six_periods:
            //following is for Morning
            Morning_num_vehicles = Morning/100 * num_vehicles
            Morning_interval = (Morning_end_time - Morning_start_time) / Morning_num_vehicles
            for (i = 0, i < Morning_num_vehicles, i++):
              departure_times.append(Morning_start_time + i*Morning_interval)
            //do the same for Morning Rush, Noon, Evening Rush, Evening, Night
            return departure_times

       if departure_pattern starts with "custom:":
            // Parse custom pattern: "custom:HH:MM-HH:MM,percent;HH:MM-HH:MM,percent;..."
            custom_windows = parse_custom_windows(departure_pattern)
            rest_windows = compute_rest_windows_custom(custom_windows, start_time, end_time) // gaps between specified windows
            rest_total_time = compute from rest_windows
            rest_percent = 100 - sum(custom_window_percents)

            for each rest_window:
              rest_window_num_vehicles = (rest_window_duration / rest_total_time) * (rest_percent/100) * num_vehicles
              rest_window_interval = rest_window_duration / rest_window_num_vehicles
              for (i = 0, i < rest_window_num_vehicles, i++):
                departure_times.append(rest_window_start_time + i*rest_window_interval)

            for each custom_window:
              window_num_vehicles = (custom_window_percent/100) * num_vehicles
              window_interval = (window_end_time - window_start_time) / window_num_vehicles
              for (i = 0, i < window_num_vehicles, i++):
                departure_times.append(window_start_time + i*window_interval)

            return departure_times
```

#### Implementing generate_public_vehicle_routes.

For puclic vehicles we need to set the route before setting the departure times

SECONDS_IN_DAY = 86400
DEFAULT_VEHICLES_DAILY_PER_ROUTE = 124

ideal_num_vehicles_per_route = (total_duration_in_secs / SECONDS_IN_DAY) \* DEFAULT_VEHICLES_DAILY_PER_ROUTE
num_public_routes = top(num_public_vehicles / ideal_num_vehicles_per_route)
num_vehicles_per_route = floor(num_public_vehicles / num_public_routes)

number of public routes from each type:
num_in_routes = (public-routes.in/100) x num_public_routes
num_out_routes = public-routes.out/100 x num_public_routes
num_inner_routes = public-routes.inner/100 x num_public_routes
num_pass_routes = public-routes.pass/100 x num_public_routes

//departure times are udentical for all route
departure_times = calculate_temporal_departure_times(num_vehicles_per_route, departure_pattern, start_time, end_time)

There might be a setting where there are not enough public vehicles to generate all the types (in/out/inner/pass) of required routes. The reason behind it is because we require several vehicles in each public route. For example if we have --public-routes "in 50 out 30 inner 10 pass 10" but because of not many public vehicles num_public_routes is 1 (or even less than 1). In this case we'll just create 1 public in route since 'in' has the highest precentage. Maybe if such cases come up we should print a warning explaining the problem. If there are not enough public vehicles then the last route should have less than ideal_num_vehicles_per_route of vehicles.

It is important the for public routes it always uses the shortest path no matter of the routing strategies.
