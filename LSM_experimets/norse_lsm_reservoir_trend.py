import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random
import seaborn as sns
import json
import os
from datetime import datetime

class LSMReservoir:
    def __init__(self, n_neurons=64, connectivity=0.5, input_size=2):
        """
        Initialize LSM reservoir with log-normal weight distribution
        n_neurons: number of neurons in the reservoir
        connectivity: probability of connection between neurons
        input_size: dimension of input (2 channels: positive and negative trends)
        """
        self.n_neurons = n_neurons
        
        # LIF neuron parameters
        tau_mem = 0.250
        self.p = norse.LIFParameters(
            tau_mem_inv=torch.tensor(1/tau_mem),
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(0.05),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(50.0)
        )
        
        # Create neurons
        self.neurons = [norse.LIFCell(self.p) for _ in range(n_neurons)]
        
        # Initialize weights with log-normal distribution
        self.reservoir_weights = self._initialize_weights(n_neurons, n_neurons, connectivity, scale=0.3)
        self.input_weights = self._initialize_weights(input_size, n_neurons, 0.9, scale=0.5)  # Higher input connectivity
        
        # Initialize neuron states
        self.states = [None] * n_neurons
        
        # Storage for activity metrics
        self.activity_history = []
        self.active_neuron_count = []
        
        # Print initialization summary
        print(f"LSM Reservoir initialized with log-normal weights:")
        print(f"- {n_neurons} neurons with {connectivity:.1%} connectivity")
        print(f"- Input weights: shape={self.input_weights.shape}, mean={torch.mean(self.input_weights):.4f}")
        
        # Print sum of weights for each input channel
        for i in range(input_size):
            channel_sum = torch.sum(self.input_weights[i]).item()
            print(f"  - Channel {i} weight sum: {channel_sum:.4f}")
        
        print(f"- Reservoir weights: shape={self.reservoir_weights.shape}, mean={torch.mean(self.reservoir_weights):.4f}")
    
    def _initialize_weights(self, rows, cols, density, scale=0.3):
        """
        Initialize weights with log-normal distribution
        
        Parameters:
        - rows, cols: dimensions of the weight matrix
        - density: probability of connection between neurons
        - scale: scaling factor for the weights
        
        Returns:
        - weights: tensor of shape (rows, cols) with log-normal weights
        """
        # Create connectivity mask
        mask = torch.rand(rows, cols) < density
        
        # Generate log-normal weights
        mu, sigma = -0.5, 0.5  # Parameters for log-normal distribution
        weights = torch.exp(torch.randn(rows, cols) * sigma + mu)
        
        # Scale weights
        weights = weights * scale / weights.mean()
        
        # Apply connectivity mask
        weights = weights * mask
        
        # Ensure max weight is at most 1.0
        if torch.max(weights) > 1.0:
            weights = weights / torch.max(weights)
            
        return weights
    
    def reset_states(self):
        """Reset all neuron states"""
        self.states = [None] * self.n_neurons
    
    def simulate(self, input_spikes, record_interval=20):
        """
        Simulate reservoir with input
        input_spikes: tensor of shape (timesteps, input_size)
        record_interval: store activity metrics every record_interval steps
        
        Returns:
        - reservoir_spikes_history: tensor of shape (timesteps, n_neurons)
        - membrane_history: tensor of shape (timesteps, n_neurons)
        """
        print(f"Starting simulation with input shape: {input_spikes.shape}")
        timesteps = input_spikes.shape[0]
        
        # Storage for spikes and membrane potentials
        reservoir_spikes_history = torch.zeros(timesteps, self.n_neurons)
        membrane_history = torch.zeros(timesteps, self.n_neurons)
        
        # Clear activity history
        self.activity_history = []
        self.active_neuron_count = []
        
        for t in range(timesteps):
            if t % 1000 == 0:
                print(f"Processing timestep {t}/{timesteps}")
            
            # Process input
            current_input = input_spikes[t].reshape(1, -1)
            input_current = torch.matmul(current_input, self.input_weights).squeeze()
            
            # Process reservoir
            reservoir_spikes = torch.zeros(self.n_neurons)
            for i in range(self.n_neurons):
                total_current = input_current[i] + torch.sum(self.reservoir_weights[:, i] * reservoir_spikes)
                out, self.states[i] = self.neurons[i](torch.tensor([total_current]), self.states[i])
                reservoir_spikes[i] = 1.0 if out.item() > 0 else 0.0
                membrane_history[t, i] = self.states[i].v.item() if self.states[i] is not None else 0.0
            
            reservoir_spikes_history[t] = reservoir_spikes
            
            # Record activity metrics
            if t % record_interval == 0:
                self.activity_history.append(torch.mean(reservoir_spikes).item())
                self.active_neuron_count.append(torch.sum(reservoir_spikes > 0).item())
        
        print("Simulation completed")
        return reservoir_spikes_history, membrane_history
    
    def visualize_weights(self):
        """Visualize weight matrices and distributions"""
        # Reservoir weights
        plt.figure(figsize=(15, 10))
        
        # Plot reservoir weight matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(self.reservoir_weights, cmap='viridis', 
                   vmin=0, vmax=max(0.3, torch.max(self.reservoir_weights).item()))
        plt.title('Reservoir Weight Matrix')
        plt.xlabel('Post-synaptic Neuron')
        plt.ylabel('Pre-synaptic Neuron')
        
        # Plot reservoir weight distribution
        weights = self.reservoir_weights.numpy().flatten()
        non_zero_weights = weights[weights > 0.01]  # Exclude near-zero weights
        
        plt.subplot(2, 2, 2)
        sns.histplot(data=non_zero_weights, bins=30, kde=True)
        plt.title('Reservoir Weight Distribution (Log-Normal)')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
        # Add statistics
        if len(non_zero_weights) > 0:
            plt.text(0.05, 0.95, 
                    f"Mean: {np.mean(non_zero_weights):.3f}\nStd: {np.std(non_zero_weights):.3f}\nMax: {np.max(non_zero_weights):.3f}", 
                    transform=plt.gca().transAxes, verticalalignment='top')
        
        # Plot input weight matrix
        plt.subplot(2, 2, 3)
        sns.heatmap(self.input_weights.detach().numpy(), cmap='viridis',
                   vmin=0, vmax=max(0.5, torch.max(self.input_weights).item()))
        plt.title('Input Weights')
        plt.xlabel('Reservoir Neuron')
        plt.ylabel('Input Channel')
        
        # Plot input weight distribution
        input_weights_flat = self.input_weights.detach().numpy().flatten()
        non_zero_input = input_weights_flat[input_weights_flat > 0.01]
        
        plt.subplot(2, 2, 4)
        sns.histplot(data=non_zero_input, bins=30, kde=True)
        plt.title('Input Weight Distribution (Log-Normal)')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
        # Add statistics including channel sums
        if len(non_zero_input) > 0:
            channel_sums = [torch.sum(self.input_weights[i]).item() for i in range(self.input_weights.shape[0])]
            stats_text = f"Mean: {np.mean(non_zero_input):.3f}\nStd: {np.std(non_zero_input):.3f}\nMax: {np.max(non_zero_input):.3f}"
            
            for i, sum_val in enumerate(channel_sums):
                stats_text += f"\nChannel {i} sum: {sum_val:.3f}"
                
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_activity(self):
        """Visualize network activity metrics"""
        if len(self.activity_history) == 0:
            print("No activity data available. Run simulation first.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot mean activity
        plt.subplot(2, 1, 1)
        plt.plot(self.activity_history)
        plt.title('Mean Network Activity')
        plt.xlabel('Time (record intervals)')
        plt.ylabel('Mean Activity (spikes/neuron)')
        plt.grid(True, alpha=0.3)
        
        # Plot active neuron count
        plt.subplot(2, 1, 2)
        plt.plot(self.active_neuron_count)
        plt.title('Number of Active Neurons')
        plt.xlabel('Time (record intervals)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load market data
    print("Loading market data...")
    with open("LSM_experimets/market_data.json", "r") as file:
        market_data = json.load(file)
    
    # Convert market data into structured lists
    dates = sorted(market_data.keys())
    trends = np.array([market_data[date]['trend'] for date in dates])
    
    # Create input spike trains from trends
    print("Creating spike trains from trends...")
    timesteps = len(trends)
    input_spikes = torch.zeros(timesteps, 2)  # Two channels: [positive_trend, negative_trend]
    
    # Parameters for spike probability conversion
    trend_threshold = 100  # Base threshold
    
    # Generate spikes based on continuous probabilities
    strong_trend_days = 0
    for t in range(timesteps):
        trend = trends[t]
        
        # Calculate probabilities for both channels
        if trend > 0:
            # Positive trend: affects first channel
            prob = min(1.0, (trend / trend_threshold) ** 2)  # Quadratic scaling
            if np.random.random() < prob:
                input_spikes[t, 0] = 1.0
                if trend > trend_threshold:
                    strong_trend_days += 1
        else:
            # Negative trend: affects second channel
            prob = min(1.0, (abs(trend) / trend_threshold) ** 2)  # Quadratic scaling
            if np.random.random() < prob:
                input_spikes[t, 1] = 1.0
                if abs(trend) > trend_threshold:
                    strong_trend_days += 1
    
    print(f"Total timesteps: {timesteps}")
    print(f"Days with strong trends (>|{trend_threshold}|): {strong_trend_days} ({(strong_trend_days/timesteps)*100:.2f}%)")
    print(f"Channel 1 (positive) spike rate: {torch.mean(input_spikes[:, 0]):.4f}")
    print(f"Channel 2 (negative) spike rate: {torch.mean(input_spikes[:, 1]):.4f}")
    
    # Create and simulate reservoir
    print("\nCreating reservoir...")
    reservoir = LSMReservoir(n_neurons=128, connectivity=0.5)
    

    
    print("Simulating reservoir...")
    reservoir_spikes, membranes = reservoir.simulate(input_spikes)
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original trends
    plt.subplot(4, 1, 1)
    plt.plot(trends, label='Market Trends')
    plt.axhline(y=trend_threshold, color='g', linestyle='--', alpha=0.5, label='Upper Threshold')
    plt.axhline(y=-trend_threshold, color='r', linestyle='--', alpha=0.5, label='Lower Threshold')
    plt.title('Market Trends')
    plt.xlabel('Time (days)')
    plt.ylabel('Trend Strength')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Input spikes (both channels)
    plt.subplot(4, 1, 2)
    plt.plot(input_spikes[:, 0].numpy(), label='Positive Trend Channel', color='green', alpha=0.7)
    plt.plot(input_spikes[:, 1].numpy(), label='Negative Trend Channel', color='red', alpha=0.7)
    plt.title('Input Spike Trains')
    plt.xlabel('Time (days)')
    plt.ylabel('Spike Value')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Reservoir response
    plt.subplot(4, 1, 3)
    mean_activity = torch.mean(reservoir_spikes, dim=1)
    plt.plot(mean_activity.numpy(), label='Mean Reservoir Activity')
    plt.title('Reservoir Response')
    plt.xlabel('Time (days)')
    plt.ylabel('Mean Activity')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Membrane potentials (sample of neurons)
    plt.subplot(4, 1, 4)
    # Select a few random neurons to display
    np.random.seed(42)  # For reproducibility
    sample_neurons = np.random.choice(range(reservoir.n_neurons), size=3, replace=False)
    for idx, neuron_idx in enumerate(sample_neurons):
        plt.plot(membranes[:, neuron_idx].numpy(), 
                label=f'Neuron #{neuron_idx}', alpha=0.7)
    plt.title('Reservoir Membrane Potentials (Sample Neurons)')
    plt.xlabel('Time (days)')
    plt.ylabel('Membrane Potential')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize weights and activity
    print("\nVisualizing weights...")
    reservoir.visualize_weights()
    
    print("\nVisualizing activity...")
    reservoir.visualize_activity()
    
    # Print output statistics
    print("\nReservoir Statistics (128 neurons):")
    print(f"Total input spikes: {torch.sum(input_spikes)}")
    print(f"Total reservoir spikes: {torch.sum(reservoir_spikes)}")
    print(f"Mean reservoir activity: {torch.mean(reservoir_spikes):.4f}")
    print(f"Mean membrane potential: {torch.mean(membranes):.4f}")
    
    # Calculate activity per neuron
    spikes_per_neuron = torch.sum(reservoir_spikes) / reservoir.n_neurons
    print(f"Average spikes per neuron: {spikes_per_neuron:.2f}")
    
    # Print input channel weight sums
    print("\nInput Channel Weight Sums:")
    for i in range(reservoir.input_weights.shape[0]):
        channel_sum = torch.sum(reservoir.input_weights[i]).item()
        print(f"Channel {i} weight sum: {channel_sum:.4f}")
    
    # Calculate neuron activity statistics
    neuron_spike_counts = torch.sum(reservoir_spikes, dim=0)
    most_active_neuron = torch.argmax(neuron_spike_counts).item()
    least_active_neuron = torch.argmin(neuron_spike_counts).item()
    
    print(f"\nNeuron Activity Analysis:")
    print(f"Most active neuron: #{most_active_neuron} with {neuron_spike_counts[most_active_neuron]} spikes")
    print(f"Least active neuron: #{least_active_neuron} with {neuron_spike_counts[least_active_neuron]} spikes")
    print(f"Neurons with zero spikes: {torch.sum(neuron_spike_counts == 0).item()} out of {reservoir.n_neurons}")
    print(f"Percentage of active neurons: {(1 - torch.sum(neuron_spike_counts == 0).item() / reservoir.n_neurons) * 100:.2f}%")
    print(f"Average spikes per neuron: {torch.mean(neuron_spike_counts):.2f}") 