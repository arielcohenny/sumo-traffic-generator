# Vehicle Route Generation

## Purpose and Process Overview

**Purpose of Vehicle Route Generation:**

- **Traffic Demand Realization**: Converts edge attractiveness values into actual vehicle trips with specific origins, destinations, and departure times
- **Realistic Vehicle Mix**: Creates diverse vehicle fleet reflecting real-world traffic composition
- **Routing Behavior**: Assigns different routing strategies to vehicles to simulate realistic navigation patterns
- **Temporal Distribution**: Distributes vehicle departures over time according to realistic daily patterns

## Vehicle Route Generation Process

- **Step**: Generate complete vehicle route definitions for simulation
- **Function**: `generate_vehicle_routes()` in `src/traffic/builder.py`
- **Arguments Used**: `--num_vehicles`, `--vehicle_types`, `--departure_pattern`, `--routing_strategy`, `--seed`, `--end_time`
- **Input Files**:
  - `workspace/grid.net.xml` (network with attractiveness attributes)
  - Uses edge attractiveness values for origin/destination selection
  - Network topology for route calculation

### Vehicle Type System (3-Type Classification)

**Vehicle Type Distribution (`--vehicle_types` parameter):**

**1. Passenger Vehicles (Default: 60%)**

- **Vehicle Class**: Personal cars, sedans, SUVs
- **SUMO Definition**: `vClass="passenger"`
- **Physical Characteristics**:
  - Length: 5.0 meters
  - Max Speed: 13.9 m/s (50 km/h)
  - Acceleration: 2.6 m/s²
  - Deceleration: 4.5 m/s²
  - Sigma (driver imperfection): 0.5
- **Behavior**: Most common vehicle type, represents private transportation

**2. Commercial Vehicles (Default: 30%)**

- **Vehicle Class**: Delivery trucks, freight vehicles, commercial vans
- **SUMO Definition**: `vClass="truck"`
- **Physical Characteristics**:
  - Length: 12.0 meters
  - Max Speed: 10.0 m/s (36 km/h)
  - Acceleration: 1.3 m/s²
  - Deceleration: 4.0 m/s²
  - Sigma (driver imperfection): 0.5
- **Behavior**: Larger, slower vehicles representing freight and delivery traffic

**3. Public Transportation (Default: 10%)**

- **Vehicle Class**: Buses, public transit vehicles
- **SUMO Definition**: `vClass="bus"`
- **Physical Characteristics**:
  - Length: 10.0 meters
  - Max Speed: 11.1 m/s (40 km/h)
  - Acceleration: 1.2 m/s²
  - Deceleration: 4.0 m/s²
  - Sigma (driver imperfection): 0.5
- **Behavior**: Public transit vehicles with specific operating characteristics

**Vehicle Type Validation:**

- **Percentage Sum**: Must total exactly 100%
- **Format**: "passenger 70 commercial 20 public 10"
- **Assignment**: Each generated vehicle randomly assigned type based on percentages

### Departure Pattern System

**Departure Pattern Distribution (`--departure_pattern` parameter):**

**1. Six Periods Pattern (Default: "six_periods")**

- **Research Basis**: Based on established 6-period daily traffic structure
- **Time Periods**:
  - Morning (6:00-7:30): 20% of daily traffic
  - Morning Rush (7:30-9:30): 30% of daily traffic
  - Noon (9:30-16:30): 25% of daily traffic
  - Evening Rush (16:30-18:30): 20% of daily traffic
  - Evening (18:30-22:00): 4% of daily traffic
  - Night (22:00-6:00): 1% of daily traffic
- **Distribution**: Vehicles assigned departure times within periods based on percentages

**2. Uniform Pattern ("uniform")**

- **Distribution**: Even distribution across entire simulation time
- **Calculation**: `departure_time = random_uniform(0, end_time)`
- **Use Case**: Baseline comparison without temporal bias

**3. Custom Rush Hours Pattern ("rush_hours:7-9:40,17-19:30,rest:10")**

- **Format**: Defines specific rush hour periods with percentages, remainder distributed to other times
- **Flexibility**: Allows custom peak periods for specific scenarios

**4. Granular Hourly Pattern ("hourly:7:25,8:35,rest:5")**

- **Format**: Assigns specific percentages to individual hours
- **Control**: Fine-grained temporal control for detailed analysis

### Routing Strategy System (4-Strategy Classification)

**Routing Strategy Assignment (`--routing_strategy` parameter):**

**1. Shortest Path Strategy ("shortest")**

- **Algorithm**: Static shortest path calculation at route generation time
- **Behavior**: Vehicles follow pre-calculated shortest routes without dynamic updates
- **Characteristics**: Fastest route calculation, no simulation-time overhead
- **Use Case**: Baseline routing without real-time adaptation

**2. Realtime Strategy ("realtime")**

- **Algorithm**: Simulates GPS navigation apps like Waze/Google Maps with dynamic rerouting every 30 seconds
- **Initial Route**: Uses fastest path algorithm, falls back to shortest path if fastest fails
- **Behavior**: Vehicles adapt routes based on current traffic conditions using frequent updates
- **Implementation**: TraCI-based route updates prioritizing responsiveness to traffic changes
- **Characteristics**: High update frequency (30s) for maximum traffic responsiveness

**3. Fastest Path Strategy ("fastest")**

- **Algorithm**: Pure travel-time optimization with dynamic rerouting every 45 seconds
- **Initial Route**: Uses fastest path algorithm exclusively (optimizes for travel time over distance)
- **Behavior**: Vehicles consistently seek minimum travel time routes regardless of distance
- **Implementation**: Less frequent but more computation-intensive rerouting focused on time efficiency
- **Characteristics**: Moderate update frequency (45s) with consistent time-optimization focus

**4. Attractiveness Strategy ("attractiveness")**

- **Algorithm**: Multi-criteria routing considering both travel time and edge attractiveness
- **Behavior**: Vehicles prefer routes through higher-attractiveness areas
- **Implementation**: Custom routing algorithm incorporating attractiveness weights
- **Characteristics**: Simulates preference for main roads or commercial areas

**Strategy Mixing:**

- **Format**: "shortest 70 realtime 30" assigns 70% shortest, 30% realtime
- **Validation**: Percentages must sum to 100%
- **Assignment**: Each vehicle randomly assigned strategy based on percentages

### Origin-Destination Selection

**Attractiveness-Based Selection:**

- **Departure Edges**: Selected based on `depart_attractiveness` values (weighted random selection)
- **Arrival Edges**: Selected based on `arrive_attractiveness` values (weighted random selection)
- **Spatial Distribution**: Higher attractiveness values increase selection probability
- **Route Feasibility**: Ensures valid routes exist between selected origin-destination pairs

## Vehicle Route Generation Validation

- **Step**: Verify route generation was successful
- **Function**: `verify_generate_vehicle_routes()` in `src/validate/validate_traffic.py`
- **Validation Checks**:
  - **Vehicle Count**: Confirm total vehicles matches `--num_vehicles` parameter
  - **Route Structure**: Verify all routes have valid origin and destination edges
  - **Type Distribution**: Check vehicle type percentages match `--vehicle_types` specification
  - **Departure Timing**: Validate departure times follow `--departure_pattern` distribution
  - **Strategy Assignment**: Confirm routing strategies match `--routing_strategy` percentages
  - **XML Validity**: Ensure output file is valid SUMO route XML format

## Vehicle Route Generation Completion

- **Step**: Confirm successful route generation
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Generated vehicle routes successfully."
- **Output File**: `workspace/vehicles.rou.xml` with complete route definitions for all vehicles
- **Ready for Next Step**: Vehicle routes are prepared for SUMO configuration generation