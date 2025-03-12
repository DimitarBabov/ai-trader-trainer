import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random

class LSMTransformer(nn.Module):
    """
    A hybrid model combining a Liquid State Machine (LSM) reservoir with a Transformer encoder.
    The LSM processes the input sequence through a spiking neural network,
    and the Transformer processes the resulting reservoir states.
    """
    def __init__(self, 
                 input_size=1, 
                 reservoir_size=100, 
                 connectivity=0.2,
                 transformer_dim=64, 
                 nhead=4, 
                 num_layers=2,
                 dropout=0.1):
        """
        Initialize the LSMTransformer model.
        
        Args:
            input_size: Dimension of input features
            reservoir_size: Number of neurons in the LSM reservoir
            connectivity: Probability of connection between neurons in the reservoir
            transformer_dim: Dimension of the Transformer model
            nhead: Number of heads in the multi-head attention
            num_layers: Number of Transformer encoder layers
            dropout: Dropout probability in the Transformer
        """
        super(LSMTransformer, self).__init__()
        
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        
        # LIF neuron parameters
        tau_mem = 0.250
        self.p = norse.LIFParameters(
            tau_mem_inv=torch.tensor(1/tau_mem),
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(0.1),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(50.0)
        )
        
        # Initialize reservoir weights (sparse random connectivity)
        density = connectivity
        reservoir_weights = random(reservoir_size, reservoir_size, density=density)
        reservoir_weights = torch.tensor(reservoir_weights.toarray(), dtype=torch.float32)
        self.register_buffer('reservoir_weights', reservoir_weights)
        
        # Input weights (fully connected)
        self.input_weights = nn.Parameter(torch.randn(input_size, reservoir_size) * 0.1)
        
        # Readout from reservoir to transformer
        self.readout = nn.Linear(reservoir_size, transformer_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=nhead, 
            dim_feedforward=transformer_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize neuron state
        self.reset_state()
    
    def reset_state(self):
        """Reset the state of all neurons in the reservoir."""
        self.state = norse.LIFState(
            v=torch.zeros(self.reservoir_size),
            i=torch.zeros(self.reservoir_size),
            z=torch.zeros(self.reservoir_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the LSMTransformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Transformer encoder output of shape (batch_size, seq_length, transformer_dim)
        """
        batch_size, seq_length, _ = x.shape
        
        # Storage for reservoir states
        reservoir_states = torch.zeros(batch_size, seq_length, self.reservoir_size)
        
        # Process each sequence in the batch
        for b in range(batch_size):
            # Reset reservoir state for each sequence
            self.reset_state()
            
            # Process each timestep
            for t in range(seq_length):
                # Get input at current timestep
                x_t = x[b, t]
                
                # Input current
                input_current = torch.matmul(x_t, self.input_weights)
                
                # Recurrent current
                if t > 0:
                    recurrent_current = torch.matmul(self.state.z, self.reservoir_weights)
                    total_current = input_current + recurrent_current
                else:
                    total_current = input_current
                
                # Update neuron state
                self.state, spikes = norse.LIFCell.apply(
                    total_current, self.state, self.p
                )
                
                # Store reservoir state (spikes)
                reservoir_states[b, t] = self.state.z
        
        # Transform reservoir states to transformer dimension
        transformer_input = self.readout(reservoir_states)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(transformer_input)
        
        return transformer_output
    
    def visualize_reservoir(self, input_signal, title="LSM Reservoir Activity"):
        """
        Visualize the reservoir activity for a given input signal.
        
        Args:
            input_signal: Input tensor of shape (seq_length, input_size)
            title: Title for the plot
        """
        with torch.no_grad():
            # Expand input to batch size 1
            x = input_signal.unsqueeze(0)  # (1, seq_length, input_size)
            
            seq_length = x.shape[1]
            
            # Reset reservoir state
            self.reset_state()
            
            # Storage for reservoir states and membrane potentials
            spikes = torch.zeros(seq_length, self.reservoir_size)
            membranes = torch.zeros(seq_length, self.reservoir_size)
            
            # Process each timestep
            for t in range(seq_length):
                # Get input at current timestep
                x_t = x[0, t]
                
                # Input current
                input_current = torch.matmul(x_t, self.input_weights)
                
                # Recurrent current
                if t > 0:
                    recurrent_current = torch.matmul(self.state.z, self.reservoir_weights)
                    total_current = input_current + recurrent_current
                else:
                    total_current = input_current
                
                # Update neuron state
                self.state, _ = norse.LIFCell.apply(
                    total_current, self.state, self.p
                )
                
                # Store reservoir state
                spikes[t] = self.state.z
                membranes[t] = self.state.v
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot input signal
            axes[0].plot(input_signal.numpy())
            axes[0].set_title("Input Signal")
            axes[0].set_ylabel("Amplitude")
            
            # Plot spike raster
            spike_indices = torch.nonzero(spikes)
            if len(spike_indices) > 0:
                axes[1].scatter(spike_indices[:, 0], spike_indices[:, 1], marker='|', s=10)
            axes[1].set_title("Spike Raster")
            axes[1].set_ylabel("Neuron Index")
            
            # Plot membrane potentials for a subset of neurons
            num_neurons_to_plot = min(10, self.reservoir_size)
            for i in range(num_neurons_to_plot):
                axes[2].plot(membranes[:, i], label=f"Neuron {i}")
            axes[2].set_title("Membrane Potentials")
            axes[2].set_xlabel("Time Step")
            axes[2].set_ylabel("Membrane Potential")
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()

# Example usage
if __name__ == "__main__":
    # Create a sine wave input
    timesteps = 100
    t = torch.linspace(0, 4*np.pi, timesteps)
    input_signal = torch.sin(t).unsqueeze(1)  # Shape: (timesteps, 1)
    
    # Create model
    model = LSMTransformer(
        input_size=1,
        reservoir_size=50,  # Smaller reservoir for visualization
        connectivity=0.2,
        transformer_dim=32,
        nhead=4,
        num_layers=2
    )
    
    # Visualize reservoir activity
    model.visualize_reservoir(input_signal, title="LSM Response to Sine Wave")
    
    # Test forward pass
    batch_size = 5
    batch_input = input_signal.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, timesteps, 1)
    output = model(batch_input)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, timesteps, transformer_dim) 