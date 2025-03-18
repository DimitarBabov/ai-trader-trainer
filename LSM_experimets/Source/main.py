"""
Main Script for Market Trend Neural Network
=========================================

This script orchestrates the simulation of a spiking neural network
that processes market trend data.
"""

import numpy as np
import os
from data_loader import load_market_data
from network import ReservoirNetwork
from visualization import plot_network_activity, print_statistics, plot_weight_matrix

def main():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to data file (one directory up)
    data_file = os.path.join(os.path.dirname(current_dir), 'market_data.json')
    
    # Load and preprocess market data
    dates, trend_values, price_values, input_signal_pos, input_signal_neg, input_spikes_pos, input_spikes_neg = load_market_data(data_file)
    
    # Create and simulate network
    network = ReservoirNetwork()
    
    # Plot weight matrix
    network_params = network.get_parameters()
    input_weights = np.zeros((2, network_params['num_neurons']))  # Create input weights matrix
    for i in range(network_params['num_neurons']):
        input_type, scale = network_params['input_connections'][i]
        channel_idx = 0 if input_type == 'pos' else 1
        input_weights[channel_idx, i] = scale
    
    # Plot weight matrix and show it
    plot_weight_matrix(network_params['cross_weights'], input_weights)
    
    # Run simulation
    membrane_potentials, spikes = network.simulate(input_signal_pos, input_signal_neg)
    
    # Create time points for plotting
    time_points = np.arange(0, len(dates), 1)
    
    # Visualize results and show them
    plot_network_activity(
        time_points=time_points,
        membrane_potentials=membrane_potentials,
        trend_values=trend_values,
        input_spikes_pos=input_spikes_pos,
        input_spikes_neg=input_spikes_neg,
        dates=dates,
        network_params=network_params
    )
    
    # Print statistics
    print_statistics(network_params)

if __name__ == "__main__":
    main()
    