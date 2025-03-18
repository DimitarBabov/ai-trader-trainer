"""
Four-Neuron Reservoir with Market Trend Data
=========================================

This script implements a four-neuron reservoir using the Norse library.
The reservoir receives two inputs (positive and negative trends) and has
all-to-all connectivity between neurons.

Key Features:
------------
1. Four LIF neurons with all-to-all connectivity
2. Two input channels:
   - Positive trends
   - Negative trends
3. Fixed synaptic weights for:
   - Input connections
   - Inter-neuron connections
4. Visualization of:
   - Network activity (membrane potentials)
   - Market trends and prices
   - Input spike patterns

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

# Create four LIF neurons
neurons = [norse.LIFCell(p) for _ in range(4)]

# Load market data and create spike train input
with open('LSM_experimets/market_data.json', 'r') as f:
    market_data = json.load(f)

# Extract trend values and limit to 1000 points
dates = list(market_data.keys())
trend_values = [market_data[date]['trend'] for date in dates]
price_values = [market_data[date]['price'] for date in dates]
timesteps = 1000
trend_values = trend_values[:timesteps]
price_values = price_values[:timesteps]
dates = dates[:timesteps]

# Create input spikes based on trend values
input_spikes_pos = np.zeros(timesteps)  # For positive trends
input_spikes_neg = np.zeros(timesteps)  # For negative trends

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
    trend = trend_values[t]
    if trend > 0:  # Positive trend
        prob = sigmoid(trend, x0, k)
        if np.random.random() < prob:
            input_spikes_pos[t] = 1.0
    else:  # Negative trend
        prob = sigmoid(-trend, x0, k)  # Use absolute value for probability
        if np.random.random() < prob:
            input_spikes_neg[t] = 1.0

# Convert to PyTorch tensors
input_signal_pos = torch.tensor(input_spikes_pos, dtype=torch.float32).reshape(timesteps, 1)
input_signal_neg = torch.tensor(input_spikes_neg, dtype=torch.float32).reshape(timesteps, 1)

# Initialize storage
states = [None] * 4
membrane_potentials = [[] for _ in range(4)]
spikes = [[] for _ in range(4)]

# Fixed weights
input_weight = 1.0  # Weight for market trend inputs
cross_weight = 0.3  # Weight for inter-neuron connections (reduced due to more connections)

# Input connectivity pattern (which neurons receive which inputs)
# First two neurons receive positive trend, last two receive negative trend
input_connections = {
    0: ('pos', 1.0),   # (input_type, weight)
    1: ('pos', 0.8),
    2: ('neg', 1.0),
    3: ('neg', 0.8)
}

# Simulate the reservoir over time
for t in range(timesteps):
    # Get previous spikes for all neurons
    prev_spikes = [spikes[i][-1] if t > 0 and spikes[i] else 0 for i in range(4)]
    
    # Process each neuron
    for i in range(4):
        # Reset state if there was a spike
        if t > 0 and prev_spikes[i] == 1:
            states[i] = None
        
        # Calculate input from other neurons
        cross_input = sum(cross_weight * spike for j, spike in enumerate(prev_spikes) if j != i)
        
        # Add market trend input based on neuron's input connectivity
        input_type, input_scale = input_connections[i]
        if input_type == 'pos':
            market_input = input_signal_pos[t] * input_weight * input_scale
        else:  # neg
            market_input = input_signal_neg[t] * input_weight * input_scale
        
        # Combine inputs
        total_input = market_input + cross_input
        
        # Simulate neuron
        out, states[i] = neurons[i](total_input, states[i])
        
        # Store results
        membrane_potentials[i].append(states[i].v.item())
        spikes[i].append(1 if out.item() > 0 else 0)

# Convert to numpy arrays
membrane_potentials = [np.array(pot) for pot in membrane_potentials]
spikes = [np.array(sp) for sp in spikes]

# Create time array for x-axis
time_points = np.arange(0, timesteps, 1)

# Plot results with shared x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 2, 1], sharex=True)

# Plot membrane potentials (combined)
for i, potentials in enumerate(membrane_potentials):
    ax1.plot(time_points, potentials, label=f"Neuron {i+1}", alpha=0.7)
ax1.axhline(y=p.v_th.item(), color="black", linestyle="--", label="Firing Threshold")
ax1.axhline(y=p.v_reset.item(), color="gray", linestyle="--", label="Reset Potential")
ax1.set_ylabel("Membrane Potential")
ax1.set_title(f"Four-Neuron Reservoir Response to Market Trends")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Plot trend values
ax2.plot(time_points, trend_values, label="Trend", color="purple", linewidth=2)
ax2.axhline(y=x0, color="red", linestyle="--", label=f"50% Spike Probability ({x0})")
ax2.axhline(y=-x0, color="blue", linestyle="--", label=f"-50% Spike Probability (-{x0})")
ax2.set_ylabel("Trend Value")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Plot input spikes only
ax3.eventplot([np.where(input_spikes_pos > 0)[0]], colors=['lightblue'], lineoffsets=[0.5], 
              linelengths=[0.5], label='Pos. Input Spikes')
ax3.eventplot([np.where(input_spikes_neg > 0)[0]], colors=['pink'], lineoffsets=[1.5], 
              linelengths=[0.5], label='Neg. Input Spikes')
ax3.set_ylabel("Input Spikes")
ax3.set_xlabel("Time Steps")
ax3.set_yticks([0.5, 1.5])
ax3.set_yticklabels(['Pos. In', 'Neg. In'])
ax3.legend(loc='upper right')

# Set major ticks at reasonable intervals
major_step = max(1, timesteps // 10)
major_ticks = np.arange(0, timesteps + 1, major_step)
for ax in [ax1, ax2, ax3]:
    ax.set_xticks(major_ticks)
    ax.set_xticks(np.arange(0, timesteps + 1, major_step // 5), minor=True)
    ax.grid(which='minor', alpha=0.2)
    # For x-axis labels, show dates
    if ax == ax3:  # Only on bottom plot
        tick_indices = np.linspace(0, timesteps-1, min(10, timesteps)).astype(int)
        tick_dates = [dates[i] for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_dates, rotation=45)

# Print statistics
print(f"\nReservoir Statistics:")
print(f"Date range: {dates[0]} to {dates[-1]}")

for i in range(4):
    print(f"\nNeuron {i+1}:")
    print(f"Input type: {input_connections[i][0].upper()}")
    print(f"Input scale: {input_connections[i][1]:.1f}")
    print(f"Output spikes: {np.sum(spikes[i])}")
    print(f"Max membrane potential: {np.max(membrane_potentials[i]):.3f}")

print(f"\nNetwork Parameters:")
print(f"Cross-connection weight: {cross_weight}")
print(f"Base input weight: {input_weight}")

plt.tight_layout()
plt.show()
