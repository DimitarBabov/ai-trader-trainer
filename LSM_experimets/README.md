# Liquid State Machine (LSM) Experiments

This folder contains various implementations and experiments with Liquid State Machines (LSM) and spiking neural networks using the Norse library.

## Files Overview

### Basic Neuron Simulations
- **neuron_demo.py**: Basic implementation of a Leaky Integrate-and-Fire (LIF) neuron using Norse.
- **norse_neuron_basic.py**: Simple LIF neuron simulation without plasticity.
- **norse_neuron_stdp.py**: LIF neuron with Spike-Timing-Dependent Plasticity (STDP).

### Advanced LSM Implementations
- **norse_lsm_reservoir.py**: Full LSM reservoir implementation with multiple interconnected neurons and STDP learning.
- **norse_lsm_reservoir_trend.py**: LSM reservoir with log-normal weight distribution for processing market trend data.
- **LsmTransformer.py**: Hybrid model combining an LSM reservoir with a Transformer neural network.

## Dependencies

All scripts require the following Python packages:
- torch
- norse
- numpy
- matplotlib
- scipy
- seaborn (for some visualizations)
- json (for market data processing)

## Usage Examples

### Basic Neuron Simulation
```python
python neuron_demo.py
```
This will simulate a single LIF neuron with random input spikes and visualize the membrane potential and spike events.

### LSM Reservoir
```python
python norse_lsm_reservoir.py
```
This will create an LSM reservoir with multiple neurons, simulate its dynamics with random input, and analyze the weight evolution through STDP.

### Market Trend Analysis with LSM
```python
python norse_lsm_reservoir_trend.py
```
This script creates an LSM reservoir with log-normal weight distribution and processes market trend data. It visualizes the reservoir's response to market trends and provides detailed statistics about neuron activity.

### LSM-Transformer Hybrid
```python
python LsmTransformer.py
```
This will create a hybrid model combining an LSM reservoir with a Transformer encoder, and visualize the reservoir activity in response to a sine wave input.

## Key Concepts

1. **Leaky Integrate-and-Fire (LIF) Neurons**: Neurons that integrate input and fire when a threshold is reached, with a leak term that causes the membrane potential to decay over time.

2. **Spike-Timing-Dependent Plasticity (STDP)**: A biological learning mechanism where synaptic weights are adjusted based on the relative timing of pre- and post-synaptic spikes.

3. **Liquid State Machine (LSM)**: A type of reservoir computing where a recurrent neural network with fixed random weights (the "reservoir") processes input signals, and the resulting states are read out by a trainable layer.

4. **Log-Normal Weight Distribution**: A biologically plausible weight distribution where most connections are weak but a few are very strong, following a log-normal distribution.

5. **Hybrid Models**: Combinations of spiking neural networks with traditional deep learning architectures like Transformers to leverage the strengths of both approaches.

## Market Trend LSM Features

The `norse_lsm_reservoir_trend.py` script includes several advanced features:

1. **Log-Normal Weight Distribution**: Creates a more biologically plausible network with a few strong connections.

2. **Input Channel Normalization**: Ensures balanced influence between positive and negative trend channels.

3. **Detailed Visualization**:
   - Weight matrices and distributions
   - Reservoir activity over time
   - Membrane potentials of sample neurons
   - Random window visualization for detailed analysis

4. **Comprehensive Statistics**:
   - Neuron activity analysis
   - Input channel weight sums
   - Spike rate analysis
   - Active neuron percentage

5. **Scalable Architecture**: Supports different reservoir sizes (64, 128 neurons) for exploring computational capacity.

## Customization

Each script contains parameters that can be adjusted to experiment with different neuron dynamics, network topologies, and learning rules. Key parameters include:

- Neuron parameters (threshold, time constants, etc.)
- STDP parameters (learning rates, time constants)
- Network topology (size, connectivity)
- Input patterns
- Weight distribution parameters (mu, sigma for log-normal)
- Trend thresholds for market data processing

Feel free to modify these parameters to explore different behaviors of the models. 