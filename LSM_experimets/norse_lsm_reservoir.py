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
        self.n_outputs = 2
        
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
        
        # Output neuron parameters
        self.p_out = norse.LIFParameters(
            tau_mem_inv=torch.tensor(1/tau_mem),
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(0.1),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(50.0)
        )
        
        # Create neurons
        self.neurons = [norse.LIFCell(self.p) for _ in range(n_neurons)]
        self.output_neurons = [norse.LIFCell(self.p_out) for _ in range(self.n_outputs)]
        
        # Initialize weights with log-normal distribution
        self.reservoir_weights = self._initialize_weights(n_neurons, n_neurons, connectivity, scale=0.3)
        self.input_weights = self._initialize_weights(input_size, n_neurons, 0.9, scale=0.5)  # Higher input connectivity
        self.output_weights = self._initialize_weights(n_neurons, self.n_outputs, 0.8, scale=0.4)
        
        # Initialize neuron states
        self.states = [None] * n_neurons
        self.output_states = [None] * self.n_outputs
        
        # Storage for activity metrics
        self.activity_history = []
        self.active_neuron_count = []
        
        # Print initialization summary
        print(f"LSM Reservoir initialized:")
        print(f"- {n_neurons} neurons with {connectivity:.1%} connectivity")
        print(f"- Input weights: shape={self.input_weights.shape}, mean={torch.mean(self.input_weights):.4f}")
        print(f"- Reservoir weights: shape={self.reservoir_weights.shape}, mean={torch.mean(self.reservoir_weights):.4f}")
        print(f"- Output weights: shape={self.output_weights.shape}, mean={torch.mean(self.output_weights):.4f}")
    
    def _initialize_weights(self, rows, cols, density, scale=0.3):
        """Initialize weights with log-normal distribution"""
        # Create connectivity mask
        mask = random(rows, cols, density=density)
        mask = torch.tensor(mask.toarray(), dtype=torch.float32)
        
        # Generate log-normal weights
        mu, sigma = -0.5, 0.5  # Parameters for log-normal distribution
        weights = torch.exp(torch.randn(rows, cols, dtype=torch.float32) * sigma + mu)
        
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
        self.output_states = [None] * self.n_outputs
    
    def simulate(self, input_spikes, record_interval=20):
        """
        Simulate reservoir with input
        input_spikes: tensor of shape (timesteps, input_size)
        record_interval: store activity metrics every record_interval steps
        """
        print(f"Starting simulation with input shape: {input_spikes.shape}")
        timesteps = input_spikes.shape[0]
        
        # Storage for spikes and membrane potentials
        reservoir_spikes_history = torch.zeros(timesteps, self.n_neurons)
        membrane_history = torch.zeros(timesteps, self.n_neurons)
        output_spikes_history = torch.zeros(timesteps, self.n_outputs)
        output_membrane_history = torch.zeros(timesteps, self.n_outputs)
        
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
            
            # Process output neurons
            output_current = torch.matmul(reservoir_spikes, self.output_weights)
            output_spikes = torch.zeros(self.n_outputs)
            
            for i in range(self.n_outputs):
                out, self.output_states[i] = self.output_neurons[i](
                    torch.tensor([output_current[i]]), 
                    self.output_states[i]
                )
                output_spikes[i] = 1.0 if out.item() > 0 else 0.0
                output_membrane_history[t, i] = self.output_states[i].v.item() if self.output_states[i] is not None else 0.0
            
            output_spikes_history[t] = output_spikes
            
            # Record activity metrics
            if t % record_interval == 0:
                self.activity_history.append(torch.mean(reservoir_spikes).item())
                self.active_neuron_count.append(torch.sum(reservoir_spikes > 0).item())
        
        print("Simulation completed")
        return reservoir_spikes_history, membrane_history, output_spikes_history, output_membrane_history
    
    def visualize_weights(self):
        """Visualize weight matrices and distributions"""
        # Reservoir weights
        plt.figure(figsize=(15, 10))
        
        # Plot reservoir weight matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(self.reservoir_weights, cmap='viridis', 
                   vmin=0, vmax=max(1.0, torch.max(self.reservoir_weights).item()))
        plt.title('Reservoir Weight Matrix')
        plt.xlabel('Post-synaptic Neuron')
        plt.ylabel('Pre-synaptic Neuron')
        
        # Plot reservoir weight distribution
        weights = self.reservoir_weights.numpy().flatten()
        non_zero_weights = weights[weights > 0.01]  # Exclude near-zero weights
        
        plt.subplot(2, 2, 2)
        sns.histplot(data=non_zero_weights, bins=20, kde=True)
        plt.title('Reservoir Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
        # Plot input weight matrix
        plt.subplot(2, 2, 3)
        sns.heatmap(self.input_weights.detach().numpy(), cmap='viridis',
                   vmin=0, vmax=max(1.0, torch.max(self.input_weights).item()))
        plt.title('Input Weights')
        plt.xlabel('Reservoir Neuron')
        plt.ylabel('Input Channel')
        
        # Plot input weight distribution
        input_weights_flat = self.input_weights.detach().numpy().flatten()
        non_zero_input = input_weights_flat[input_weights_flat > 0.01]
        
        plt.subplot(2, 2, 4)
        sns.histplot(data=non_zero_input, bins=20, kde=True)
        plt.title('Input Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        
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
    trend_threshold = 80  # Base threshold
    
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
    reservoir = LSMReservoir(n_neurons=64, connectivity=0.5)
    
    print("Simulating reservoir...")
    reservoir_spikes, membranes, output_spikes, output_membranes = reservoir.simulate(input_spikes)
    
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
    
    # Plot 4: Output spike trains
    plt.subplot(4, 1, 4)
    plt.plot(output_spikes[:, 0].numpy(), label='Output 1', alpha=0.7)
    plt.plot(output_spikes[:, 1].numpy(), label='Output 2', alpha=0.7)
    plt.title('Output Spike Trains')
    plt.xlabel('Time (days)')
    plt.ylabel('Spike Value')
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
    print("\nOutput Statistics:")
    print(f"Output 1 spike rate: {torch.mean(output_spikes[:, 0]):.4f}")
    print(f"Output 2 spike rate: {torch.mean(output_spikes[:, 1]):.4f}")
    print(f"Total output 1 spikes: {torch.sum(output_spikes[:, 0])}")
    print(f"Total output 2 spikes: {torch.sum(output_spikes[:, 1])}")
    
    # Save daily trend values and output spikes to a text file
    print("\nSaving daily trend values and output spikes to a text file...")
    
    # Create output directory if it doesn't exist
    output_dir = "LSM_experimets/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/daily_trends_and_spikes_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        # Write header
        f.write("Date,Trend,Input_Spike_Pos,Input_Spike_Neg,Output_Spike_1,Output_Spike_2,Reservoir_Activity\n")
        
        # Write data for each day
        for i in range(len(dates)):
            date = dates[i]
            trend = trends[i]
            input_pos = input_spikes[i, 0].item()
            input_neg = input_spikes[i, 1].item()
            output_1 = output_spikes[i, 0].item()
            output_2 = output_spikes[i, 1].item()
            res_activity = mean_activity[i].item()
            
            f.write(f"{date},{trend:.2f},{input_pos:.0f},{input_neg:.0f},{output_1:.0f},{output_2:.0f},{res_activity:.4f}\n")
    
    print(f"Data saved to {output_file}")
    
    # Additional analysis: Count days with output spikes
    days_with_output_1 = torch.sum(output_spikes[:, 0] > 0).item()
    days_with_output_2 = torch.sum(output_spikes[:, 1] > 0).item()
    days_with_any_output = torch.sum(torch.max(output_spikes, dim=1)[0] > 0).item()
    days_with_both_outputs = torch.sum(torch.min(output_spikes, dim=1)[0] > 0).item()
    
    print("\nOutput Spike Analysis:")
    print(f"Days with Output 1 spikes: {days_with_output_1} ({days_with_output_1/timesteps*100:.2f}%)")
    print(f"Days with Output 2 spikes: {days_with_output_2} ({days_with_output_2/timesteps*100:.2f}%)")
    print(f"Days with any output spike: {days_with_any_output} ({days_with_any_output/timesteps*100:.2f}%)")
    print(f"Days with both outputs spiking: {days_with_both_outputs} ({days_with_both_outputs/timesteps*100:.2f}%)") 