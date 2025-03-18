"""
Neural Network Module
===================

This module defines the neural network architecture and simulation logic
for the spiking neural network using Norse.
"""

import torch
import norse.torch as norse
import numpy as np
from typing import List, Tuple, Dict
import os
from visualization import plot_state_vector

class ReservoirNetwork:
    def __init__(self, num_neurons: int = 128, input_weight: float = 0.5, use_seed: bool = False):
        """
        Initialize the reservoir network.
        
        Args:
            num_neurons: Total number of neurons in the reservoir (default: 128)
            input_weight: Scaling factor for input weights
            use_seed: Whether to use a fixed random seed for reproducibility
        """
        self.num_neurons = num_neurons
        self.input_weight = input_weight
        
        if use_seed:
            np.random.seed(42)
        
        # Initialize cross-connection weights with log-normal distribution
        self.cross_weights = self._initialize_weights(
            self.num_neurons, 
            self.num_neurons, 
            density=0.5,  # 50% connectivity
            scale=0.3
        )
        
        # Randomly select 30% of neurons from each group to connect to the input
        pos_neurons = np.random.choice(range(num_neurons // 2), size=int(num_neurons * 0.3 // 2), replace=False)
        neg_neurons = np.random.choice(range(num_neurons // 2, num_neurons), size=int(num_neurons * 0.3 // 2), replace=False)
        
        # Initialize input connections with balanced weights
        self.input_connections = [(None, 0.0)] * num_neurons
        
        # Generate weights for positive trend neurons
        pos_weights = np.random.rand(len(pos_neurons))
        pos_weights = pos_weights / np.sum(pos_weights)  # Normalize to sum to 1
        pos_weights = pos_weights * input_weight * len(pos_neurons)  # Scale by input_weight and number of neurons
        
        # Generate weights for negative trend neurons
        neg_weights = np.random.rand(len(neg_neurons))
        neg_weights = neg_weights / np.sum(neg_weights)  # Normalize to sum to 1
        neg_weights = neg_weights * input_weight * len(neg_neurons)  # Scale by input_weight and number of neurons
        
        # Assign weights
        for i, neuron_idx in enumerate(pos_neurons):
            self.input_connections[neuron_idx] = ('pos', pos_weights[i])
        for i, neuron_idx in enumerate(neg_neurons):
            self.input_connections[neuron_idx] = ('neg', neg_weights[i])
        
        # Define LIF neuron parameters
        self.p = norse.LIFParameters(
            tau_mem_inv=torch.tensor(1/0.150),
            v_leak=torch.tensor(0.00),
            v_th=torch.tensor(0.1),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(50.0)
        )
        
        # Initialize neurons
        self.neurons = [norse.LIFCell(self.p) for _ in range(num_neurons)]
        
        # Create state_images directory if it doesn't exist
        self.state_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'state_images')
        os.makedirs(self.state_images_dir, exist_ok=True)
    
    def _initialize_weights(self, rows: int, cols: int, density: float, scale: float = 0.3) -> np.ndarray:
        """
        Initialize weights with sparse connectivity, respecting positive/negative trend separation
        
        Parameters:
        - rows, cols: dimensions of the weight matrix
        - density: probability of connection between neurons
        - scale: scaling factor for the weights
        
        Returns:
        - weights: numpy array of shape (rows, cols) with sparse log-normal weights
        """
        # Create base weight matrix
        weights = np.zeros((rows, cols))
        
        # Split neurons into positive and negative trend groups
        half_size = rows // 2
        
        # Generate connections within each group
        for group_start in [0, half_size]:
            group_end = group_start + half_size
            
            # Generate connections for this group
            for i in range(group_start, group_end):
                # Create connection mask for this group
                mask = np.zeros(rows)
                # Allow connections to all neurons in both groups
                mask[:] = np.random.rand(rows) < density
                
                # Generate log-normal weights
                mu, sigma = -0.5, 0.5
                neuron_weights = np.exp(np.random.normal(mu, sigma, rows))
                
                # Scale weights
                neuron_weights = neuron_weights * scale / neuron_weights.mean()
                
                # Apply mask and store in weight matrix
                weights[i] = neuron_weights * mask
        
        # Ensure max weight is at most 1.0
        if np.max(weights) > 1.0:
            weights = weights / np.max(weights)
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(weights, 0)
        
        # Make the weight matrix symmetric
        weights = (weights + weights.T) / 2
            
        return weights
    
    def simulate(self, input_signal_pos: torch.Tensor, input_signal_neg: torch.Tensor) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Simulate the reservoir over time.
        
        Args:
            input_signal_pos: Positive trend input spikes
            input_signal_neg: Negative trend input spikes
            
        Returns:
            tuple: (membrane_potentials, spikes)
        """
        timesteps = len(input_signal_pos)
        print(f"Starting simulation...")
        
        # Initialize storage
        states = [None] * self.num_neurons
        membrane_potentials = [[] for _ in range(self.num_neurons)]
        spikes = [[] for _ in range(self.num_neurons)]
        
        # Storage for first 1000 states
        # state_history = []
        # spike_history = []
        
        # Statistics tracking
        last_spike_time = [0] * self.num_neurons  # Track last spike time for each neuron
        spike_intervals = [[] for _ in range(self.num_neurons)]  # Store intervals between spikes
        spike_counts = [0] * self.num_neurons  # Count total spikes per neuron
        
        # Simulate over time
        for t in range(timesteps):
            if t % 1000 == 0 and t > 0:
                print(f"Processed {t} steps...")
            
            # Get previous spikes for all neurons
            prev_spikes = [spikes[i][-1] if t > 0 and spikes[i] else 0 for i in range(self.num_neurons)]
            
            # Process each neuron
            current_potentials = []
            current_spikes = []
            for i in range(self.num_neurons):
                # Reset state if there was a spike
                if t > 0 and prev_spikes[i] == 1:
                    states[i] = None
                
                # Calculate input from other neurons using random weights
                cross_input = sum(self.cross_weights[i, j] * prev_spikes[j] for j in range(self.num_neurons))
                
                # Add market trend input based on neuron's input connectivity
                input_type, input_scale = self.input_connections[i]
                if input_type == 'pos':
                    market_input = input_signal_pos[t] * self.input_weight * input_scale
                else:  # neg
                    market_input = input_signal_neg[t] * self.input_weight * input_scale
                
                # Combine inputs
                total_input = market_input + cross_input
                
                # Simulate neuron
                out, states[i] = self.neurons[i](total_input, states[i])
                spike = 1 if out.item() > 0 else 0
                potential = states[i].v.item() if states[i] is not None else 0.0
                
                # Track spike statistics
                if spike == 1:
                    if last_spike_time[i] > 0:  # Not the first spike
                        interval = t - last_spike_time[i]
                        spike_intervals[i].append(interval)
                    last_spike_time[i] = t
                    spike_counts[i] += 1
                
                # Store results
                membrane_potentials[i].append(potential)
                spikes[i].append(spike)
                current_potentials.append(potential)
                current_spikes.append(spike)
            
            # Store state for visualization if in first 1000 steps
            # if t < 1000:
            #     state_history.append(current_potentials)
            #     spike_history.append(current_spikes)
        
        print("Simulation completed.")
        
        # Save all state images in batch
        # for t in range(len(state_history)):
        #     save_path = os.path.join(self.state_images_dir, f'state_step_{t:04d}.png')
        #     plot_state_vector(state_history[t], spike_history[t], save_path, t)
        
        # print("State images saved")
        
        # Calculate and print firing statistics
        print("\nFiring Statistics:")
        print("-" * 50)
        
        # Group neurons by input type
        pos_neurons = [i for i in range(self.num_neurons) if self.input_connections[i][0] == 'pos']
        neg_neurons = [i for i in range(self.num_neurons) if self.input_connections[i][0] == 'neg']
        
        # Calculate statistics for positive trend neurons
        pos_intervals = [interval for i in pos_neurons for interval in spike_intervals[i]]
        pos_counts = [spike_counts[i] for i in pos_neurons]
        print(f"Positive Trend Neurons (0-{self.num_neurons//2-1}):")
        print(f"  Total spikes: {sum(pos_counts)}")
        print(f"  Average spikes per neuron: {np.mean(pos_counts):.2f}")
        if pos_intervals:
            print(f"  Average firing interval: {np.mean(pos_intervals):.2f} steps")
            print(f"  Min firing interval: {min(pos_intervals)} steps")
            print(f"  Max firing interval: {max(pos_intervals)} steps")
        
        # Calculate statistics for negative trend neurons
        neg_intervals = [interval for i in neg_neurons for interval in spike_intervals[i]]
        neg_counts = [spike_counts[i] for i in neg_neurons]
        print(f"\nNegative Trend Neurons ({self.num_neurons//2}-{self.num_neurons-1}):")
        print(f"  Total spikes: {sum(neg_counts)}")
        print(f"  Average spikes per neuron: {np.mean(neg_counts):.2f}")
        if neg_intervals:
            print(f"  Average firing interval: {np.mean(neg_intervals):.2f} steps")
            print(f"  Min firing interval: {min(neg_intervals)} steps")
            print(f"  Max firing interval: {max(neg_intervals)} steps")
        
        # Convert to numpy arrays
        membrane_potentials = [np.array(pot) for pot in membrane_potentials]
        spikes = [np.array(sp) for sp in spikes]
        
        return membrane_potentials, spikes
    
    def get_parameters(self) -> Dict:
        """
        Get network parameters.
        
        Returns:
            dict: Dictionary containing network parameters
        """
        return {
            'num_neurons': self.num_neurons,
            'input_weight': self.input_weight,
            'cross_weights': self.cross_weights,
            'threshold': self.p.v_th.item(),
            'reset': self.p.v_reset.item(),
            'input_connections': self.input_connections
        } 