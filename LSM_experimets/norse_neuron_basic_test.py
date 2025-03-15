"""
Basic Two-Neuron Reservoir with Market Trend Data
==============================================

This script implements a simple two-neuron reservoir using the Norse library.
One neuron responds to positive trends, the other to negative trends, and they
are interconnected with fixed weights.

Key Features:
------------
1. Two LIF neurons: Each neuron responds to different trend directions
   - Neuron 1: Responds to positive trends
   - Neuron 2: Responds to negative trends

2. Interconnected neurons: Each neuron receives input from the other's previous spike

3. Market trend input: Uses trend data to generate spikes based on sigmoid
   probability for each trend direction.

4. Detailed visualization: Plots membrane potentials, trends, and spikes
   for both neurons.

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

# Create two LIF neurons
lif_cell_pos = norse.LIFCell(p)  # For positive trends
lif_cell_neg = norse.LIFCell(p)  # For negative trends

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
state_pos = None
state_neg = None
membrane_potentials_pos = []
membrane_potentials_neg = []
spikes_pos = []
spikes_neg = []

# Fixed weights
input_weight = 1.0  # Weight for market trend inputs
cross_weight = 0.5  # Weight for cross-neuron connections

# Simulate the reservoir over time
for t in range(timesteps):
    # Get cross-inputs from previous timestep
    cross_input_pos = cross_weight * (spikes_neg[-1] if t > 0 else 0)
    cross_input_neg = cross_weight * (spikes_pos[-1] if t > 0 else 0)
    
    # Reset states if there was a spike
    if t > 0:
        if spikes_pos[-1] == 1:
            state_pos = None
        if spikes_neg[-1] == 1:
            state_neg = None
    
    # Combine market input with cross-neuron input
    weighted_input_pos = input_signal_pos[t] * input_weight + cross_input_pos
    weighted_input_neg = input_signal_neg[t] * input_weight + cross_input_neg
    
    # Simulate neurons
    out_pos, state_pos = lif_cell_pos(weighted_input_pos, state_pos)
    out_neg, state_neg = lif_cell_neg(weighted_input_neg, state_neg)
    
    # Store results
    membrane_potentials_pos.append(state_pos.v.item())
    membrane_potentials_neg.append(state_neg.v.item())
    spikes_pos.append(1 if out_pos.item() > 0 else 0)
    spikes_neg.append(1 if out_neg.item() > 0 else 0)

# Convert to numpy arrays
membrane_potentials_pos = np.array(membrane_potentials_pos)
membrane_potentials_neg = np.array(membrane_potentials_neg)
spikes_pos = np.array(spikes_pos)
spikes_neg = np.array(spikes_neg)

# Create time array for x-axis
time_points = np.arange(0, timesteps, 1)

# Plot results with shared x-axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16), height_ratios=[2, 2, 2, 1], sharex=True)

# Plot membrane potentials
ax1.plot(time_points, membrane_potentials_pos, label="Pos. Neuron Membrane", color="blue", linewidth=2)
ax1.plot(time_points, membrane_potentials_neg, label="Neg. Neuron Membrane", color="red", linewidth=2)
ax1.axhline(y=p.v_th.item(), color="black", linestyle="--", label="Firing Threshold")
ax1.axhline(y=p.v_reset.item(), color="gray", linestyle="--", label="Reset Potential")
ax1.set_ylabel("Membrane Potential")
ax1.set_title(f"Two-Neuron Reservoir Response to Market Trends")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Plot trend values
ax2.plot(time_points, trend_values, label="Trend", color="purple", linewidth=2)
ax2.axhline(y=x0, color="red", linestyle="--", label=f"50% Spike Probability ({x0})")
ax2.axhline(y=-x0, color="blue", linestyle="--", label=f"-50% Spike Probability (-{x0})")
ax2.set_ylabel("Trend Value")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Plot price data
ax3.plot(time_points, price_values, label="Price", color="green", linewidth=2)
ax3.set_ylabel("Price")
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Plot spikes for both neurons
ax4.eventplot([np.where(input_spikes_pos > 0)[0]], colors=['lightblue'], lineoffsets=[0.5], 
              linelengths=[0.5], label='Pos. Input Spikes')
ax4.eventplot([np.where(input_spikes_neg > 0)[0]], colors=['pink'], lineoffsets=[1.0], 
              linelengths=[0.5], label='Neg. Input Spikes')
ax4.eventplot([np.where(spikes_pos > 0)[0]], colors=['blue'], lineoffsets=[1.5], 
              linelengths=[0.5], label='Pos. Neuron Spikes')
ax4.eventplot([np.where(spikes_neg > 0)[0]], colors=['red'], lineoffsets=[2.0], 
              linelengths=[0.5], label='Neg. Neuron Spikes')
ax4.set_ylabel("Spike Events")
ax4.set_xlabel("Time Steps")
ax4.set_yticks([0.5, 1.0, 1.5, 2.0])
ax4.set_yticklabels(['Pos. In', 'Neg. In', 'Pos. Out', 'Neg. Out'])
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

# Print statistics
print(f"\nReservoir Statistics:")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"\nPositive Trend Neuron:")
print(f"Input spikes: {np.sum(input_spikes_pos > 0)}")
print(f"Output spikes: {np.sum(spikes_pos > 0)}")
print(f"Max membrane potential: {np.max(membrane_potentials_pos):.3f}")

print(f"\nNegative Trend Neuron:")
print(f"Input spikes: {np.sum(input_spikes_neg > 0)}")
print(f"Output spikes: {np.sum(spikes_neg > 0)}")
print(f"Max membrane potential: {np.max(membrane_potentials_neg):.3f}")

print(f"\nCross-connection weight: {cross_weight}")
print(f"Input weight: {input_weight}")

plt.tight_layout()
plt.show()
