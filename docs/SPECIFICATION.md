# SUMO Traffic Generator Specification

This document provides the formal specification for the SUMO Traffic Generator, detailing all available CLI arguments and the 9-step pipeline execution process.

## Project Overview

The SUMO Traffic Generator is a sophisticated Python-based framework that creates dynamic traffic simulations with intelligent signal control. It provides both command-line and web-based GUI interfaces for configuration and execution. The system supports synthetic orthogonal grid networks and Tree Method research datasets, applies configurable lane assignments, and uses Tree Method's decentralized traffic control algorithm for dynamic signal optimization.

## Table of Contents

### Running and Interfaces

- **[Command Line Interface](specification/command-line-interface.md)** - CLI arguments and execution
- **[Web GUI Interface](specification/web-gui-interface.md)** - Streamlit-based graphical interface
- **[SUMO GUI Integration](specification/sumo-gui-integration.md)** - Built-in SUMO visualization

### Pipeline Steps

- **[Initialization and Setup](specification/initialization-setup.md)** - Command-line arguments, seed management, and validation
- **[Network Generation](specification/network-generation.md)** - Grid creation and Tree Method samples
- **[Zone Generation](specification/zone-generation.md)** - Land use zone extraction for synthetic grids
- **[Edge Splitting and Lane Assignment](specification/edge-splitting-lanes.md)** - Traffic flow optimization algorithms
- **[Network Rebuild](specification/network-rebuild.md)** - SUMO network compilation process
- **[Edge Attractiveness Assignment](specification/edge-attractiveness.md)** - Traffic demand modeling methods
- **[Vehicle Route Generation](specification/vehicle-route-generation.md)** - Traffic pattern creation and routing strategies
- **[SUMO Configuration Generation](specification/sumo-configuration.md)** - Simulation setup and file coordination
- **[Dynamic Simulation with Traffic Control](specification/dynamic-simulation.md)** - Execution engine and Tree Method algorithm

### Quality Assurance

- **[Validation](specification/validation.md)** - Runtime validation framework and error handling
- **[Testing Framework](specification/testing-framework.md)** - Software testing infrastructure and procedures
- **[Scripts](specification/scripts.md)** - Development utilities and testing tools

### Research Framework

- **[Experimental Evaluation Framework](specification/experimental-framework.md)** - Statistical validation and benchmarking system