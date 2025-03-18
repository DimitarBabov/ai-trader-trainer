"""
Visualization Module
==================

This module handles all visualization functions for the neural network,
including membrane potentials, trends, and spike patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import os

def plot_weight_matrix(
    cross_weights: np.ndarray,
    input_weights: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot the weight matrices and their distributions.
    
    Args:
        cross_weights: 2D numpy array of reservoir weights
        input_weights: 2D numpy array of input weights
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot reservoir weight matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cross_weights, cmap='viridis', 
                vmin=0, vmax=1.0,
                xticklabels=4, yticklabels=4)
    plt.title('Reservoir Weight Matrix')
    plt.xlabel('Post-synaptic Neuron')
    plt.ylabel('Pre-synaptic Neuron')
    
    # Plot reservoir weight distribution
    weights = cross_weights.flatten()
    non_zero_weights = weights[weights > 0.01]
    
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
    sns.heatmap(input_weights, cmap='viridis',
                vmin=0, vmax=1.0)
    plt.title('Input Weights')
    plt.xlabel('Reservoir Neuron')
    plt.ylabel('Input Channel')
    
    # Plot input weight distribution
    input_weights_flat = input_weights.flatten()
    non_zero_input = input_weights_flat[input_weights_flat > 0.01]
    
    plt.subplot(2, 2, 4)
    sns.histplot(data=non_zero_input, bins=30, kde=True)
    plt.title('Input Weight Distribution (Log-Normal)')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    # Add statistics including channel sums
    if len(non_zero_input) > 0:
        channel_sums = [np.sum(input_weights[i]) for i in range(input_weights.shape[0])]
        stats_text = f"Mean: {np.mean(non_zero_input):.3f}\nStd: {np.std(non_zero_input):.3f}\nMax: {np.max(non_zero_input):.3f}"
        for i, sum_val in enumerate(channel_sums):
            stats_text += f"\nChannel {i} sum: {sum_val:.3f}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_state_vector(
    membrane_potentials: List[float],
    spikes: List[int],
    save_path: str,
    step: int
) -> None:
    """
    Plot and save the state vector of the network at a given step as an 8x8 grayscale image.
    
    Args:
        membrane_potentials: List of membrane potentials for each neuron
        spikes: List of spike values (0 or 1) for each neuron
        save_path: Path to save the image
        step: Current simulation step
    """
    # Reshape membrane potentials into 8x8 grid
    potentials = np.array(membrane_potentials).reshape(8, 8)
    
    # Create figure with just the image
    plt.figure(figsize=(8, 8))
    
    # Plot as grayscale image
    plt.imshow(potentials, cmap='gray', vmin=0, vmax=0.1)
    
    # Remove axes
    plt.axis('off')
    
    # Add step number as title
    plt.title(f"Step {step}", pad=20)
    
    # Save with tight layout
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_network_activity(
    time_points: np.ndarray,
    membrane_potentials: List[np.ndarray],
    trend_values: List[float],
    input_spikes_pos: np.ndarray,
    input_spikes_neg: np.ndarray,
    dates: List[str],
    network_params: Dict,
    save_path: str = None
) -> None:
    """
    Plot network activity including membrane potentials, trends, and spikes.
    """
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Market Trends
    plt.subplot(4, 1, 1)
    plt.plot(trend_values, label='Market Trends', color='blue')
    plt.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Upper Threshold')
    plt.axhline(y=-100, color='r', linestyle='--', alpha=0.5, label='Lower Threshold')
    plt.title('Market Trends')
    plt.xlabel('Time (days)')
    plt.ylabel('Trend Strength')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Input Spike Trains
    plt.subplot(4, 1, 2)
    plt.plot(input_spikes_pos, label='Positive Trend Channel', color='green', alpha=0.7)
    plt.plot(input_spikes_neg, label='Negative Trend Channel', color='red', alpha=0.7)
    plt.title('Input Spike Trains')
    plt.xlabel('Time (days)')
    plt.ylabel('Spike Value')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Reservoir Response (Mean Activity)
    plt.subplot(4, 1, 3)
    mean_activity = np.mean([np.array(mp) for mp in membrane_potentials], axis=0)
    plt.plot(mean_activity, label='Mean Reservoir Activity')
    plt.title('Reservoir Response')
    plt.xlabel('Time (days)')
    plt.ylabel('Mean Activity')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Sample Neuron Membrane Potentials
    plt.subplot(4, 1, 4)
    np.random.seed(42)  # For reproducibility
    sample_neurons = np.random.choice(range(len(membrane_potentials)), size=3, replace=False)
    for idx, neuron_idx in enumerate(sample_neurons):
        plt.plot(membrane_potentials[neuron_idx], 
                label=f'Neuron #{neuron_idx}', alpha=0.7)
    plt.title('Reservoir Membrane Potentials (Sample Neurons)')
    plt.xlabel('Time (days)')
    plt.ylabel('Membrane Potential')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_activity_statistics(
    activity_history: List[float],
    active_neuron_count: List[int],
    save_path: str = None
) -> None:
    """
    Plot network activity statistics.
    
    Args:
        activity_history: List of mean activity values
        active_neuron_count: List of active neuron counts
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot mean activity
    plt.subplot(2, 1, 1)
    plt.plot(activity_history)
    plt.title('Mean Network Activity')
    plt.xlabel('Time (record intervals)')
    plt.ylabel('Mean Activity (spikes/neuron)')
    plt.grid(True)
    
    # Plot active neuron count
    plt.subplot(2, 1, 2)
    plt.plot(active_neuron_count)
    plt.title('Number of Active Neurons')
    plt.xlabel('Time (record intervals)')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def print_statistics(network_params):
    """
    Print statistics of the network parameters.
    
    Args:
        network_params: Dictionary containing network parameters
    """
    print("\nNetwork Parameters:")
    print("-" * 50)
    print(f"Number of neurons: {network_params['num_neurons']}")
    print(f"Input weight: {network_params['input_weight']}")
    print(f"Threshold: {network_params['threshold']:.3f}")
    print(f"Reset potential: {network_params['reset']:.3f}")
    
    # Print input connections
    print("\nInput Connections:")
    print("-" * 50)
    
    # Count neurons by type
    pos_count = 0
    neg_count = 0
    unconnected = 0
    
    for i, (input_type, input_scale) in enumerate(network_params['input_connections']):
        if input_type == 'pos':
            pos_count += 1
            print(f"Neuron {i:3d}: Input type: POS, Scale: {input_scale:.3f}")
        elif input_type == 'neg':
            neg_count += 1
            print(f"Neuron {i:3d}: Input type: NEG, Scale: {input_scale:.3f}")
        else:
            unconnected += 1
    
    print("\nConnectivity Summary:")
    print("-" * 50)
    print(f"Positive trend neurons: {pos_count}")
    print(f"Negative trend neurons: {neg_count}")
    print(f"Unconnected neurons: {unconnected}")
    print(f"Total neurons: {network_params['num_neurons']}") 