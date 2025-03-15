"""
Basic Leaky Integrate-and-Fire (LIF) Neuron Simulation with Market Trend Data
==========================================================================

This script implements a basic Leaky Integrate-and-Fire (LIF) neuron using the Norse library
for spiking neural networks. It uses market trend data to generate input spikes based on
a quadratic probability threshold.

Key Features:
------------
1. Single LIF neuron: Implements a single spiking neuron with configurable parameters
   such as membrane time constant, threshold, and reset potential.

2. Market trend input: Uses trend data to generate spikes based on a quadratic
   probability threshold when trend > 100.

3. Fixed synaptic weight: Uses a constant weight (no plasticity) to demonstrate
   the basic input-output relationship of the neuron.

4. Detailed visualization: Plots membrane potential, input spikes, and output spikes
   to illustrate the neuron's dynamics.

Usage:
------
Run this script to observe how a LIF neuron responds to market trend-based spikes
with a fixed synaptic weight. The visualization shows the membrane potential evolution
and the relationship between input and output spikes.

Implementation Details:
----------------------
- LIF parameters: Configurable parameters for the neuron model
- Market data processing: Converts trend values to spikes using quadratic probability
- Simulation loop: Processes input spikes and updates neuron state
- Visualization: Plots membrane potential and spike events
- Statistics: Calculates and displays spike statistics

Dependencies:
------------
- torch: For tensor operations
- norse: For spiking neural network implementation
- numpy: For numerical operations
- matplotlib: For visualization
- json: For loading market data
"""

import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import matplotlib.pyplot as plt
import json

# Define LIF neuron parameters
tau_mem = 0.150  # Fast membrane time constant
p = norse.LIFParameters(
    tau_mem_inv=torch.tensor(1/tau_mem),  # Convert time constant to inverse
    v_leak=torch.tensor(0.00),
    v_th=torch.tensor(0.1),              # Threshold
    v_reset=torch.tensor(0.0),
    method="super",
    alpha=torch.tensor(50.0)             # Increased smoothing factor
)
lif_cell = norse.LIFCell(p)

# Load market data and create spike train input
with open('LSM_experimets/market_data.json', 'r') as f:
    market_data = json.load(f)

# Extract trend values and limit to 500 points
dates = list(market_data.keys())
trend_values = [market_data[date]['trend'] for date in dates]
price_values = [market_data[date]['price'] for date in dates]  # Extract price data
timesteps = 1000  # Use only first 1500 points
trend_values = trend_values[:timesteps]
price_values = price_values[:timesteps]  # Limit price data
dates = dates[:timesteps]

# Create input spikes based on trend values
input_spikes = np.zeros(timesteps)

# Define sigmoid probability function parameters
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

# Parameters tuned to give:
# ~10% probability at trend=30
# ~50% probability at trend=70
x0 = 70  # center point (50% probability)
k = 0.05  # steepness

# Convert trend values to spikes using sigmoid probability
for t in range(timesteps):
    trend = abs(trend_values[t])  # Use absolute trend value
    prob = sigmoid(trend, x0, k)
    if np.random.random() < prob:
        input_spikes[t] = 1.0

# Print probability examples for verification
print("\nSpike Probability Examples:")
print(f"At trend=30: {sigmoid(30, x0, k)*100:.1f}%")
print(f"At trend=70: {sigmoid(70, x0, k)*100:.1f}%")
print(f"At trend=90: {sigmoid(90, x0, k)*100:.1f}%")
print(f"At trend=120: {sigmoid(120, x0, k)*100:.1f}%")

# Convert to PyTorch tensor
input_signal = torch.tensor(input_spikes, dtype=torch.float32).reshape(timesteps, 1)

# Initialize storage
state = None
membrane_potentials = []
spikes = []

# Fixed input weight (no plasticity)
weight = 1.0

# Simulate the LIF neuron over time
for t in range(timesteps):
    # If we just had a spike, reset the state completely
    if t > 0 and spikes[-1] == 1:
        state = None
    
    # Apply fixed weight to input
    weighted_input = input_signal[t] * weight
    
    # Simulate neuron
    out, state = lif_cell(weighted_input, state)
    membrane_potentials.append(state.v.item())
    spike = 1 if out.item() > 0 else 0
    spikes.append(spike)

# Convert to numpy arrays
membrane_potentials = np.array(membrane_potentials)
spikes = np.array(spikes)

# Create time array for x-axis
time_points = np.arange(0, timesteps, 1)

# Plot results with shared x-axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16), height_ratios=[2, 2, 2, 1], sharex=True)

# Plot membrane potential and trend
ax1.plot(time_points, membrane_potentials, label="Membrane Potential", color="blue", linewidth=2)
ax1.axhline(y=p.v_th.item(), color="red", linestyle="--", label="Firing Threshold")
ax1.axhline(y=p.v_reset.item(), color="gray", linestyle="--", label="Reset Potential")
ax1.set_ylabel("Membrane Potential")
ax1.set_title(f"LIF Neuron Response to Market Trends")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Plot trend values
ax2.plot(time_points, trend_values, label="Trend", color="purple", linewidth=2)
ax2.axhline(y=x0, color="red", linestyle="--", label=f"50% Spike Probability ({x0})")
ax2.axhline(y=30, color="orange", linestyle="--", label="10% Spike Probability (30)")
ax2.set_ylabel("Trend Value")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Plot price data
ax3.plot(time_points, price_values, label="Price", color="green", linewidth=2)
ax3.set_ylabel("Price")
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Plot input and output spikes
ax4.eventplot([np.where(input_spikes > 0)[0]], colors=['green'], lineoffsets=[0.5], 
              linelengths=[0.5], label='Input Spikes (from Trend)')
ax4.eventplot([np.where(spikes > 0)[0]], colors=['black'], lineoffsets=[1.5], 
              linelengths=[0.5], label='Output Spikes')
ax4.set_ylabel("Spike Events")
ax4.set_xlabel("Time Steps")
ax4.set_yticks([0.5, 1.5])
ax4.set_yticklabels(['Input', 'Output'])
ax4.legend(loc='upper right')

# Set major ticks at reasonable intervals
major_step = max(1, timesteps // 10)
major_ticks = np.arange(0, timesteps + 1, major_step)
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks(major_ticks)
    ax.set_xticks(np.arange(0, timesteps + 1, major_step // 5), minor=True)
    ax.grid(which='minor', alpha=0.2)
    # For x-axis labels, show dates
    if ax == ax4:  # Only on bottom plot
        tick_indices = np.linspace(0, timesteps-1, min(10, timesteps)).astype(int)
        tick_dates = [dates[i] for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_dates, rotation=45)

# Add a text box with spike probabilities
textstr = '\n'.join((
    'Spike Probabilities:',
    f'Trend=30: {sigmoid(30, x0, k)*100:.1f}%',
    f'Trend=60: {sigmoid(60, x0, k)*100:.1f}%',
    f'Trend=90: {sigmoid(90, x0, k)*100:.1f}%'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# Print statistics
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Number of input spikes: {np.sum(input_spikes > 0)}")
print(f"Number of output spikes: {np.sum(spikes > 0)}")
print(f"Maximum membrane potential: {np.max(membrane_potentials):.3f}")
print(f"Price range: {min(price_values):.2f} to {max(price_values):.2f}")
print(f"Spike generation threshold: {x0}")

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
