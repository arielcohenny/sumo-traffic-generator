# Edge Attractiveness Assignment

## Purpose and Process Overview

**Purpose of Edge Attractiveness Assignment:**

- **Traffic Generation Foundation**: Determines how many vehicles depart from and arrive at each edge during simulation
- **Realistic Traffic Patterns**: Creates non-uniform traffic distribution that reflects real-world urban traffic flow
- **Spatial Traffic Modeling**: Accounts for land use patterns, accessibility, and network topology in traffic generation
- **Temporal Traffic Variation**: Enables time-of-day traffic patterns (rush hours, off-peak periods)

**Universal Application**: Works for both OSM and non-OSM networks after proper zone coordinate alignment

## Edge Attractiveness Assignment Process

- **Step**: Assign departure and arrival attractiveness values to all network edges
- **Function**: `assign_edge_attractiveness()` in `src/network/edge_attrs.py`
- **Arguments Used**: `--attractiveness`, `--time_dependent`, `--start_time_hour`, `--seed`
- **Input Files**:
  - `workspace/grid.net.xml` (rebuilt network with final edge definitions)
  - `workspace/zones.poly.xml` (zones with correct coordinate system from Steps 2/5)
  - `workspace/grid.edg.xml` (original edge file for spatial analysis methods)

### Five Attractiveness Calculation Methods

**1. Poisson Method (`--attractiveness poisson`)**

- **Algorithm**: Uses Poisson probability distribution to generate random attractiveness values
- **Process**: For each edge, draws random numbers from two separate Poisson distributions - one for departures (λ=3.5) and one for arrivals (λ=2.0). The Poisson distribution naturally produces positive integer values with realistic variation around the mean.
- **Mathematical Basis**: Poisson distributions model random events occurring at a constant average rate, making them suitable for modeling traffic generation where events (vehicle trips) happen independently
- **Characteristics**: Produces values clustered around the lambda parameters with occasional higher values, creating natural traffic variation without spatial bias
- **Use Case**: Baseline random traffic generation without spatial considerations
- **Independence**: No dependency on zones or network topology

**2. Land Use Method (`--attractiveness land_use`)**

- **Algorithm**: Calculates attractiveness based purely on the land use characteristics of zones adjacent to each edge
- **Process**: For each edge, identifies all zones within 10 meters using geometric intersection analysis. Calculates base attractiveness from the density and type of adjacent zones, with each zone contributing based on its attractiveness value and land use multipliers. When multiple zones are adjacent, computes weighted average where zones with higher attractiveness values have proportionally more influence. Final attractiveness values reflect pure spatial land use patterns without random components.
- **🔴 IMPLEMENTATION FIX NEEDED**: Current implementation incorrectly starts with random Poisson baseline values (`depart_base = np.random.poisson(lam=CONFIG.LAMBDA_DEPART)`) before applying land use multipliers. This introduces random variation unrelated to spatial land use patterns.
- **🔴 REQUIRED CHANGES**:
  - Remove Poisson baseline calculation from `calculate_attractiveness_land_use()` function
  - Calculate base attractiveness directly from zone density (e.g., sum of adjacent zone attractiveness values)
  - Apply land use multipliers to the zone-derived base values instead of random Poisson values
  - Ensure edges with no adjacent zones get minimal but non-zero attractiveness (e.g., default value of 1)
- **Land Use Logic**: Different land use types generate different traffic patterns - residential areas attract incoming traffic (people coming home) while employment areas generate outgoing traffic (people leaving for work). Mixed-use areas have balanced patterns.
- **Spatial Analysis**: Uses geometric intersection and distance calculations to determine edge-zone adjacency, ensuring edges near commercial districts get higher arrival attractiveness while edges near residential areas get higher departure attractiveness
- **Zone Types**: Residential, Employment, Mixed, Entertainment/Retail, Public Buildings, Public Open Space
- **Examples**:
  - Residential areas: High arrival (1.4x), moderate departure (0.8x) - people return home
  - Employment areas: High departure (1.3x), moderate arrival (0.9x) - people leave for work

**3. Gravity Method (`--attractiveness gravity`)**

- **Algorithm**: Models traffic attractiveness based on network accessibility and connectivity patterns
- **Process**: For each edge, calculates a "cluster size" by counting how many other edges connect to the same start and end nodes, representing the edge's position in the network hierarchy. Applies exponential decay with distance (using normalized distance of 1.0) and exponential growth with cluster connectivity. Multiplies by a random baseline factor to introduce variation.
- **Network Theory**: Based on gravity models from transportation planning where locations with better connectivity (more connections) attract more traffic, similar to how larger cities attract more travelers in regional models
- **Centrality Logic**: Edges connecting highly connected nodes (major intersections) get higher attractiveness than edges connecting peripheral nodes, reflecting real-world patterns where arterial roads carry more traffic than residential streets
- **Parameters**: `d_param = 0.95` (distance decay), `g_param = 1.02` (connectivity amplification)
- **Stochastic Element**: Includes random baseline factor to prevent purely deterministic results

**4. IAC Method (`--attractiveness iac`)**

- **Algorithm**: Integrated Attraction Coefficient that synthesizes multiple urban planning factors into a comprehensive attractiveness measure
- **Process**: Calculates separate gravity and land use components, then normalizes them against baseline values to create dimensionless factors. Introduces a random "mood" factor representing daily variations in travel behavior and a spatial preference factor. Multiplies all components together with a base attractiveness coefficient to produce the final IAC value.
- **Multi-Factor Integration**: Combines the network connectivity insights from gravity models with the land use patterns from zoning analysis, while accounting for behavioral unpredictability through stochastic elements
- **Research Foundation**: Based on established IAC methodology from urban traffic modeling literature that recognizes traffic generation as a function of both infrastructure (connectivity) and land use (activity patterns)
- **Normalization Strategy**: Converts gravity and land use results to relative factors (comparing against baseline Poisson values) so they can be meaningfully combined regardless of their original scales
- **Behavioral Elements**: Includes random mood factor and spatial preference to capture human decision-making variability in travel choices

**5. Hybrid Method (`--attractiveness hybrid`)**

- **Algorithm**: Combines multiple methodologies using a carefully balanced weighting scheme to capture benefits of each approach while mitigating individual weaknesses
- **Process**: Starts with pure Poisson values as the foundation, then calculates land use and gravity adjustments separately. Converts these adjustments to multiplicative factors relative to Poisson baselines, then reduces their impact (land use to 50%, gravity to 30%) to prevent any single method from dominating. Applies these dampened factors sequentially to the base Poisson values.
- **Weighting Philosophy**: Uses Poisson as the stable foundation (100% weight) because it provides consistent baseline variation. Land use gets moderate influence (50%) to incorporate spatial realism without over-constraining results. Gravity gets lighter influence (30%) to add network topology awareness without creating extreme centrality bias.
- **Robustness Strategy**: By combining methods with reduced individual impacts, the hybrid approach is less sensitive to problems in any single methodology (e.g., poor zone data or network topology issues) while still benefiting from their insights
- **Computational Balance**: Provides realistic spatial and topological variation while maintaining computational efficiency and avoiding the complexity of full IAC integration

### Temporal Variation System (4-Phase)

**Time-Dependent Mode (`--time_dependent` flag):**

**Phase Definition**:

- **Morning Peak (6:00-9:30)**: High outbound traffic (home→work), multipliers: depart 1.4x, arrive 0.7x
- **Midday Off-Peak (9:30-16:00)**: Balanced baseline traffic, multipliers: depart 1.0x, arrive 1.0x
- **Evening Peak (16:00-19:00)**: High inbound traffic (work→home), multipliers: depart 0.7x, arrive 1.5x
- **Night Low (19:00-6:00)**: Minimal activity, multipliers: depart 0.4x, arrive 0.4x

**Implementation**:

- **Base Calculation**: Generate base attractiveness using selected method without time dependency
- **Phase-Specific Profiles**: Create attractiveness values for each of 4 phases using multipliers
- **Active Phase**: Set current phase based on `--start_time_hour` parameter
- **Attribute Storage**: Store both base values and all phase-specific values in network file

### Edge Processing and Filtering

**Edge Selection**:

- **Include**: All regular network edges (roads, streets)
- **Exclude**: Internal edges (starting with ":") used for junction connections
- **Processing**: Iterate through all edges in rebuilt network file

**Attribute Assignment**:

- **Standard Mode**: `depart_attractiveness` and `arrive_attractiveness` attributes
- **Time-Dependent Mode**: Additional phase-specific attributes:
  - `morning_peak_depart_attractiveness`, `morning_peak_arrive_attractiveness`
  - `midday_offpeak_depart_attractiveness`, `midday_offpeak_arrive_attractiveness`
  - `evening_peak_depart_attractiveness`, `evening_peak_arrive_attractiveness`
  - `night_low_depart_attractiveness`, `night_low_arrive_attractiveness`

## Spatial Analysis Integration

**Zone-Edge Adjacency Detection** (for land_use, iac, hybrid methods):

- **Edge Geometry**: Parse edge shape coordinates from network
- **Zone Polygon**: Load zone polygon coordinates from zones file
- **Spatial Query**: Determine if edge intersects or is within 10 meters of zone polygon
- **Multiple Zones**: Handle cases where edges are adjacent to multiple zones
- **Weighted Calculation**: Average attractiveness based on zone attractiveness values and types

## Edge Attractiveness Validation

- **Step**: Verify attractiveness assignment was successful
- **Function**: `verify_assign_edge_attractiveness()` in `src/validate/validate_network.py`
- **Validation Checks**:
  - **Attribute Existence**: Confirm all edges have required attractiveness attributes
  - **Value Ranges**: Ensure attractiveness values are positive integers
  - **Distribution Check**: Verify reasonable distribution across network edges
  - **Time-Dependent Validation**: For temporal mode, check all phase-specific attributes exist
  - **Spatial Consistency**: Verify spatial methods produce location-appropriate values

## Edge Attractiveness Assignment Completion

- **Step**: Confirm successful attractiveness assignment
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Assigned edge attractiveness successfully."
- **Output File**: Updated `workspace/grid.net.xml` with attractiveness attributes on all edges
- **Ready for Next Step**: Network edges now have traffic generation parameters for vehicle route generation