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
    v_th=torch.tensor(0.1),              # Threshold
    v_reset=torch.tensor(0.0),
    method="super",
    alpha=torch.tensor(50.0)             # Increased smoothing factor
)
lif_cell = norse.LIFCell(p)

# Create spike train input
timesteps = 3000
input_spikes = np.zeros(timesteps)

# Always have first spike at t=0
input_spikes[0] = 1.0

# Create random input spikes with varying intervals
current_time = 0
while current_time < timesteps:
    interval = np.random.randint(2, 50)  # Random intervals between spikes
    current_time += interval
    if current_time < timesteps:
        input_spikes[current_time] = 1.0

# Convert to PyTorch tensor
input_signal = torch.tensor(input_spikes, dtype=torch.float32).reshape(timesteps, 1)

# Initialize storage
state = None
membrane_potentials = []
spikes = []

# Fixed input weight (no plasticity)
weight = 2.0

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[2, 1], sharex=True)

# Plot membrane potential and input spikes
ax1.plot(time_points, membrane_potentials, label="Membrane Potential", color="blue", linewidth=2)
ax1.axhline(y=p.v_th.item(), color="red", linestyle="--", label="Firing Threshold")
ax1.axhline(y=p.v_reset.item(), color="gray", linestyle="--", label="Reset Potential")
ax1.set_ylabel("Membrane Potential")
ax1.set_title("LIF Neuron Response (No Plasticity)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Plot input and output spikes
ax2.eventplot([np.where(input_spikes > 0)[0]], colors=['green'], lineoffsets=[0.5], 
              linelengths=[0.5], label='Input Spikes')
ax2.eventplot([np.where(spikes > 0)[0]], colors=['black'], lineoffsets=[1.5], 
              linelengths=[0.5], label='Output Spikes')
ax2.set_ylabel("Spike Events")
ax2.set_xlabel("Time Steps")
ax2.set_yticks([0.5, 1.5])
ax2.set_yticklabels(['Input', 'Output'])
ax2.legend(loc='upper right')

# Set major ticks every 100 steps and minor ticks every 20 steps
major_ticks = np.arange(0, timesteps + 1, 100)
minor_ticks = np.arange(0, timesteps + 1, 20)
for ax in [ax1, ax2]:
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)

# Print statistics
print(f"Number of input spikes: {np.sum(input_spikes > 0)}")
print(f"Number of output spikes: {np.sum(spikes > 0)}")
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
