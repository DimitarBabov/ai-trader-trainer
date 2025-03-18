"""
Leaky Integrate-and-Fire (LIF) Neuron Demonstration
==================================================

This script provides a basic demonstration of a Leaky Integrate-and-Fire (LIF) neuron
using the Norse library. It shows how a spiking neuron responds to random input spikes
and visualizes the membrane potential dynamics.

Key Features:
------------
1. Single LIF neuron: Implements a single spiking neuron with configurable parameters
   such as membrane time constant, threshold, and reset potential.

2. Random spike input: Generates a random spike train with varying intervals to
   stimulate the neuron, ensuring the first spike occurs at t=0.

3. Membrane potential visualization: Shows how the membrane potential evolves over time,
   including the effects of input spikes, leakage, and reset after firing.

4. Spike raster plot: Displays both input and output spikes to illustrate the
   relationship between input stimulation and neuron firing.

Usage:
------
Run this script to observe the basic behavior of a LIF neuron. The visualization
shows the membrane potential dynamics and the timing of input and output spikes.
This serves as an introductory demonstration of spiking neural networks.

Implementation Details:
----------------------
- LIF parameters: Configurable parameters for the neuron model
- Input generation: Creates random spike patterns with controlled intervals
- Simulation loop: Processes input spikes and updates neuron state
- Visualization: Plots membrane potential and spike events with detailed formatting
- Statistics: Calculates and displays spike statistics and intervals

Dependencies:
------------
- torch: For tensor operations
- norse: For spiking neural network implementation
- numpy: For numerical operations
- matplotlib: For visualization
"""

import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import matplotlib.pyplot as plt

# Define LIF neuron parameters
tau_mem = 0.250  # Fast membrane time constant
p = norse.LIFParameters(
    tau_mem_inv=torch.tensor(1/tau_mem),  # Convert time constant to inverse
    v_leak=torch.tensor(0.00),
    v_th=torch.tensor(0.15),              # Threshold
    v_reset=torch.tensor(0.0),
    method="super",
    alpha=torch.tensor(50.0)             # Increased smoothing factor
)
lif_cell = norse.LIFCell(p)

# Create spike train input
timesteps = 1000
input_spikes = np.zeros(timesteps)

# Always have first spike at t=0
input_spikes[0] = 1.0

# Create random input spikes with varying intervals starting after first spike
current_time = 0  # Start from 0
while current_time < timesteps:
    interval = np.random.randint(2, 100)  # Random intervals between spikes
    current_time += interval
    if current_time < timesteps:
        input_spikes[current_time] = 1.0

# Convert to PyTorch tensor
input_signal = torch.tensor(input_spikes, dtype=torch.float32).reshape(timesteps, 1) * 5.0  # Input strength

# Initialize storage
state = None
membrane_potentials = []
spikes = []

# Simulate the LIF neuron over time
for t in range(timesteps):
    # If we just had a spike, reset the state completely
    if t > 0 and spikes[-1] == 1:
        state = None
        
    out, state = lif_cell(input_signal[t], state)
    membrane_potentials.append(state.v.item())
    spikes.append(1 if out.item() > 0 else 0)

# Convert to numpy for visualization
membrane_potentials = np.array(membrane_potentials)
spikes = np.array(spikes)

# Create time array for x-axis
time_points = np.arange(0, timesteps, 1)

# Plot results with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1], sharex=True)

# Plot membrane potential and input spikes
ax1.plot(time_points, membrane_potentials, label="Membrane Potential", color="blue", linewidth=2)
ax1.axhline(y=p.v_th.item(), color="red", linestyle="--", label="Firing Threshold")
ax1.axhline(y=p.v_reset.item(), color="gray", linestyle="--", label="Reset Potential")
ax1.set_ylabel("Membrane Potential")
ax1.set_title("LIF Neuron Response to Random Input Spikes (First Spike at t=0)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Set major ticks every 100 steps and minor ticks every 20 steps
major_ticks = np.arange(0, timesteps + 1, 100)
minor_ticks = np.arange(0, timesteps + 1, 20)
ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.grid(which='minor', alpha=0.2)

# Plot input and output spikes
ax2.eventplot([np.where(input_spikes > 0)[0]], colors=['green'], lineoffsets=[0.5], 
              linelengths=[0.5], label='Input Spikes')
ax2.eventplot([np.where(spikes > 0)[0]], colors=['black'], lineoffsets=[1.5], 
              linelengths=[0.5], label='Output Spikes')
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Spike Events")
ax2.set_yticks([0.5, 1.5])
ax2.set_yticklabels(['Input', 'Output'])
ax2.legend(loc='upper right')

# Share the same x-axis ticks
ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.grid(True, alpha=0.3)
ax2.grid(which='minor', alpha=0.2)

# Print some statistics
num_input_spikes = np.sum(input_spikes > 0)
num_output_spikes = np.sum(spikes > 0)
print(f"Number of input spikes: {num_input_spikes}")
print(f"Number of output spikes: {num_output_spikes}")
print(f"Maximum membrane potential: {np.max(membrane_potentials):.3f}")

# Calculate and print inter-spike intervals if we have spikes
input_spike_times = np.where(input_spikes > 0)[0]
if len(input_spike_times) > 1:
    intervals = np.diff(input_spike_times)
    print(f"\nInput spike intervals statistics:")
    print(f"Mean interval: {np.mean(intervals):.2f}")
    print(f"Min interval: {np.min(intervals)}")
    print(f"Max interval: {np.max(intervals)}")
else:
    print("\nNot enough spikes to calculate interval statistics")

plt.tight_layout()
plt.show()
