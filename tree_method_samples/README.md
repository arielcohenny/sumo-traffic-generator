# Tree Method Sample - Original Experiment1

This directory contains one test case from original research for validating our Tree Method implementation.

## Files

- `network.net.xml` - Pre-built urban network topology (166KB)
- `vehicles.trips.xml` - Traffic demand with 946 vehicles (2.7MB) 
- `simulation.sumocfg.xml` - SUMO configuration for 2-hour simulation
- `README.md` - This documentation

## Test Case Details

**Source**: `decentralized-traffic-bottlenecks/data/Experiment1-realistic-high-load/1/`

**Scenario**: Realistic High Load
- **946 vehicles** over **2 hours** (7300 seconds)
- **Complex urban network** with multi-lane edges and traffic signals
- **Passenger vehicles only** with realistic departure patterns
- **Multiple O-D pairs** across the network

## Usage

This test case can be used to validate our Tree Method implementation against original research data.

**To run with our system:**
```bash
# Copy files to data directory and run simulation
cp tree_method_samples/*.xml data/
cd data
sumo -c simulation.sumocfg.xml
```

**Key Differences from Our Generated Networks:**
- Pre-split edges with 2-5 lanes (no edge splitting needed)
- Complex real-world topology (not synthetic grid)
- Advanced traffic signal configurations
- Higher vehicle density and routing complexity

This provides a benchmark for comparing our Tree Method performance against established research results.