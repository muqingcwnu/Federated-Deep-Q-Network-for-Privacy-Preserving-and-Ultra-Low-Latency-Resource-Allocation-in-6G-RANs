# FedDQN: Federated Deep Q-Network for 6G Resource Allocation

This repository implements a Federated Deep Q-Network (FedDQN) approach for resource allocation in 6G wireless networks. The implementation includes both centralized and federated learning approaches, along with several baseline algorithms for comparison.

## Overview

The project implements a reinforcement learning-based resource allocation system for 6G wireless networks, featuring:

- Federated Deep Q-Network (FedDQN) implementation
- Centralized DQN baseline
- DDPG baseline
- Traditional scheduling algorithms (PFS, Round Robin, Static Average)
- Comprehensive performance evaluation and visualization

## Features

- **Federated Learning**: Implements federated averaging for distributed training
- **Multiple Baselines**: Includes various baseline algorithms for comparison
- **Performance Metrics**: Tracks latency, fairness, throughput, and communication overhead
- **Visualization**: Generates detailed plots and statistical analysis
- **Configurable Parameters**: Easy-to-modify hyperparameters and network settings

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy 1.19.2+
- Matplotlib 3.5.0+
- Pandas 1.3.0+
- SciPy 1.7.0+
- Gym 0.21.0+

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── experiment.py          # Main experiment runner
├── feddqn_agent.py       # FedDQN implementation
├── baselines.py          # Baseline algorithms
├── simulation_env.py     # Network simulation environment
├── config.py            # Configuration parameters
├── requirements.txt     # Project dependencies
└── results/            # Generated plots and statistics
```

## Usage

Run the experiment:
```bash
python experiment.py
```

This will:
1. Train the FedDQN and baseline algorithms
2. Generate performance plots
3. Save statistical analysis to CSV
4. Create visualization plots in the results directory

## Results

The experiment generates several plots:
- Latency comparison
- Fairness index analysis
- Throughput distribution
- Communication overhead
- Radar plot for overall performance

All results are saved in the `results` directory, including:
- PNG and PDF versions of all plots
- Detailed statistics in `experiment_stats.csv`

## Implementation Details

### FedDQN Agent
- Implements federated learning with local DQN agents
- Uses experience replay and target networks
- Supports both centralized and federated modes

### Simulation Environment
- Models wireless network with multiple base stations
- Implements queue-based packet scheduling
- Tracks performance metrics in real-time

### Baseline Algorithms
- Centralized DQN
- DDPG
- Proportional Fair Scheduling (PFS)
- Round Robin
- Static Average


```



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
