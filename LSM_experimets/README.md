# Liquid State Machine (LSM) Experiments

This folder contains various implementations and experiments with Liquid State Machines (LSM) and spiking neural networks using the Norse library.

## Files Overview

### Basic Neuron Simulations
- **neuron_demo.py**: Basic implementation of a Leaky Integrate-and-Fire (LIF) neuron using Norse.
- **norse_neuron_basic.py**: Simple LIF neuron simulation without plasticity.
- **norse_neuron_stdp.py**: LIF neuron with Spike-Timing-Dependent Plasticity (STDP).

### Advanced LSM Implementations
- **norse_lsm_reservoir.py**: Full LSM reservoir implementation with multiple interconnected neurons and STDP learning.
- **LsmTransformer.py**: Hybrid model combining an LSM reservoir with a Transformer neural network.

## Dependencies

All scripts require the following Python packages:
- torch
- norse
- numpy
- matplotlib
- scipy
- seaborn (for some visualizations)

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

### LSM-Transformer Hybrid
```python
python LsmTransformer.py
```
This will create a hybrid model combining an LSM reservoir with a Transformer encoder, and visualize the reservoir activity in response to a sine wave input.

## Key Concepts

1. **Leaky Integrate-and-Fire (LIF) Neurons**: Neurons that integrate input and fire when a threshold is reached, with a leak term that causes the membrane potential to decay over time.

2. **Spike-Timing-Dependent Plasticity (STDP)**: A biological learning mechanism where synaptic weights are adjusted based on the relative timing of pre- and post-synaptic spikes.

3. **Liquid State Machine (LSM)**: A type of reservoir computing where a recurrent neural network with fixed random weights (the "reservoir") processes input signals, and the resulting states are read out by a trainable layer.

4. **Hybrid Models**: Combinations of spiking neural networks with traditional deep learning architectures like Transformers to leverage the strengths of both approaches.

## Customization

Each script contains parameters that can be adjusted to experiment with different neuron dynamics, network topologies, and learning rules. Key parameters include:

- Neuron parameters (threshold, time constants, etc.)
- STDP parameters (learning rates, time constants)
- Network topology (size, connectivity)
- Input patterns

Feel free to modify these parameters to explore different behaviors of the models. 