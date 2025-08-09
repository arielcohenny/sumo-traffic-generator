# SUMO Configuration Generation

## Purpose and Process Overview

**Purpose of SUMO Configuration Generation:**

- **Simulation Setup**: Creates the central configuration file that links all generated components for SUMO execution
- **File Coordination**: Ensures proper referencing of network, routes, and zones files
- **Parameter Integration**: Incorporates simulation parameters (step length, end time, GUI settings) into configuration
- **Simulation Preparation**: Final step before dynamic simulation execution

## SUMO Configuration Generation Process

- **Step**: Generate SUMO configuration file linking all simulation components
- **Function**: `generate_sumo_conf_file()` in `src/sumo_integration/sumo_utils.py`
- **Arguments Used**: `--step_length`, `--end_time`, `--gui`
- **Input Files**:
  - `workspace/grid.net.xml` (complete network with attractiveness)
  - `workspace/vehicles.rou.xml` (vehicle routes and types)
  - `workspace/zones.poly.xml` (zones for visualization)

## Configuration File Creation

**SUMO Configuration File:**

- **Output**: `workspace/grid.sumocfg` with references to all simulation components
- **Content**:
  - Network file reference: `workspace/grid.net.xml`
  - Route file reference: `workspace/vehicles.rou.xml`
  - Additional files: `workspace/zones.poly.xml`
  - Simulation parameters: step length, end time, GUI settings
- **Format**: XML configuration following SUMO standards

## SUMO Configuration Generation Completion

- **Step**: Confirm successful configuration file generation
- **Function**: Success logging in `src/cli.py`
- **Success Message**: "Generated SUMO configuration successfully."
- **Output File**: `workspace/grid.sumocfg` ready for simulation execution
- **Ready for Next Step**: All components prepared for dynamic simulation