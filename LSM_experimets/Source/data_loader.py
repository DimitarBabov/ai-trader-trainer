"""
Market Data Loader Module
========================

This module handles loading and preprocessing of market data for the neural network.
It converts market trends into spike trains using a sigmoid probability function.
"""

import json
import numpy as np
import torch

def sigmoid(x: float, x0: float, k: float) -> float:
    """
    Compute sigmoid probability for spike generation.
    
    Args:
        x: Input value
        x0: Center point (50% probability)
        k: Steepness parameter
    
    Returns:
        float: Probability value between 0 and 1
    """
    return 1 / (1 + np.exp(-k * (x - x0)))

def load_market_data(file_path: str, timesteps: int = 1000):
    """
    Load market data from JSON file and preprocess it.
    
    Args:
        file_path: Path to the JSON file containing market data
        timesteps: Number of timesteps to use
    
    Returns:
        tuple: (dates, trend_values, price_values, input_signal_pos, input_signal_neg)
    """
    # Load market data
    with open(file_path, 'r') as f:
        market_data = json.load(f)
    
    # Extract data
    dates = list(market_data.keys())[:timesteps]
    trend_values = [market_data[date]['trend'] for date in dates]
    price_values = [market_data[date]['price'] for date in dates]
    
    # Parameters for spike generation
    x0 = 70  # center point (50% probability)
    k = 0.05  # steepness
    
    # Create input spikes
    input_spikes_pos = np.zeros(timesteps)
    input_spikes_neg = np.zeros(timesteps)
    
    # Convert trend values to spikes using sigmoid probability
    for t in range(timesteps):
        trend = trend_values[t]
        if trend > 0:  # Positive trend
            prob = sigmoid(trend, x0, k)
            if np.random.random() < prob:
                input_spikes_pos[t] = 1.0
        else:  # Negative trend
            prob = sigmoid(-trend, x0, k)
            if np.random.random() < prob:
                input_spikes_neg[t] = 1.0
    
    # Convert to PyTorch tensors
    input_signal_pos = torch.tensor(input_spikes_pos, dtype=torch.float32).reshape(timesteps, 1)
    input_signal_neg = torch.tensor(input_spikes_neg, dtype=torch.float32).reshape(timesteps, 1)
    
    return dates, trend_values, price_values, input_signal_pos, input_signal_neg, input_spikes_pos, input_spikes_neg 