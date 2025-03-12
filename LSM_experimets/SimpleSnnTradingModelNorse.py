import torch
import torch.nn as nn
import norse.torch as norse
import numpy as np
import json
import matplotlib.pyplot as plt

class LSMTradingModel(nn.Module):
    def __init__(self, num_reservoir=20, p_connect=0.3):
        super().__init__()
        
        # Network architecture matching the NEST version
        self.num_inputs = 2  # trend > 100 and trend < -100
        self.num_reservoir = num_reservoir
        self.num_outputs = 2  # buy and sell signals
        
        # Input layer with lower threshold
        self.input_layer = norse.LIFCell(
            p=norse.LIFParameters(
                tau_mem_inv=1/5.0,   # Faster membrane time constant
                v_th=0.2,            # Lower threshold
                v_reset=0.0,
                alpha=100.0          # Increased smoothing factor
            )
        )
        
        # Reservoir layer
        self.reservoir_layer = norse.LIFCell(
            p=norse.LIFParameters(
                tau_mem_inv=1/3.0,   # Even faster for reservoir
                v_th=0.15,           # Lower threshold
                v_reset=0.0,
                alpha=100.0
            )
        )
        
        # Output layer
        self.output_layer = norse.LIFCell(
            p=norse.LIFParameters(
                tau_mem_inv=1/3.0,
                v_th=0.1,            # Lowest threshold for outputs
                v_reset=0.0,
                alpha=100.0
            )
        )
        
        # Initialize weights with stronger connections
        # Input to reservoir connections
        self.w_in = nn.Parameter(
            torch.rand(self.num_inputs, self.num_reservoir) * (torch.rand(self.num_inputs, self.num_reservoir) < p_connect).float() * 0.3  # Increased weight scale
        )
        
        # Reservoir to output connections
        self.w_out = nn.Parameter(
            torch.rand(self.num_reservoir, self.num_outputs) * (torch.rand(self.num_reservoir, self.num_outputs) < p_connect).float() * 0.3
        )
        
        # Initialize states
        self.reset_states()
    
    def reset_states(self):
        """Reset all neuron states"""
        self.s1 = None
        self.s2 = None
        self.s3 = None
    
    def forward(self, x):
        """Forward pass through the network"""
        # Scale up input to generate more activity
        x = x * 2.0
        
        # Input layer
        z1, self.s1 = self.input_layer(x, self.s1)
        
        # Reservoir layer
        reservoir_input = torch.mm(z1, self.w_in)
        z2, self.s2 = self.reservoir_layer(reservoir_input, self.s2)
        
        # Output layer
        output_input = torch.mm(z2, self.w_out)
        z3, self.s3 = self.output_layer(output_input, self.s3)
        
        return z3, self.s3.v if self.s3 is not None else torch.zeros_like(output_input)

def update_weights(model, reward, learning_rate=0.01):
    """Update weights using REINFORCE-like rule"""
    with torch.no_grad():
        # Scale learning rate based on reward magnitude
        scaled_lr = learning_rate * (1.0 + abs(reward) / 1000.0)  # Adaptive learning rate
        
        # Update input to reservoir weights
        delta_w_in = scaled_lr * reward * torch.randn_like(model.w_in)
        model.w_in.data += delta_w_in
        model.w_in.data.clamp_(0, 2)  # Allow slightly larger weights
        
        # Update reservoir to output weights
        delta_w_out = scaled_lr * reward * torch.randn_like(model.w_out)
        model.w_out.data += delta_w_out
        model.w_out.data.clamp_(0, 2)

def main():
    # === Load Market Data ===
    with open("LSM_experimets/market_data.json", "r") as file:
        market_data = json.load(file)
    
    # Convert JSON into structured lists
    dates = list(market_data.keys())
    prices = np.array([market_data[date]['price'] for date in dates])
    trends = np.array([market_data[date]['trend'] for date in dates])
    
    # Normalize trends to smaller range
    trends = (trends - np.mean(trends)) / (np.std(trends) + 1e-6)
    
    # === Simulation Parameters ===
    initial_funds = 10_000
    fixed_trade_size = 100
    learning_rate = 0.01
    num_episodes = 50
    
    # Initialize model
    model = LSMTradingModel()
    reward_history = []
    
    # Training loop
    for episode in range(num_episodes):
        balance = initial_funds
        stock_owned = 0
        model.reset_states()
        
        for i in range(len(prices)):
            # Convert trend to input spikes with lower thresholds
            input_spikes = torch.zeros(1, 2)
            if trends[i] > 0.5:  # Lower threshold for positive trend
                input_spikes[0, 0] = 1.0
            elif trends[i] < -0.5:  # Lower threshold for negative trend
                input_spikes[0, 1] = 1.0
            
            # Forward pass
            output_spikes, voltages = model(input_spikes)
            
            # Execute trades based on spikes and voltage levels
            if output_spikes[0, 0] > 0 or (voltages is not None and voltages[0, 0] > 0.05):  # Buy signal
                if balance >= fixed_trade_size * prices[i]:
                    balance -= fixed_trade_size * prices[i]
                    stock_owned += fixed_trade_size
            
            if output_spikes[0, 1] > 0 or (voltages is not None and voltages[0, 1] > 0.05):  # Sell signal
                if stock_owned >= fixed_trade_size:
                    balance += fixed_trade_size * prices[i]
                    stock_owned -= fixed_trade_size
        
        # Calculate final value and reward
        final_value = balance + (stock_owned * prices[-1])
        reward = final_value - initial_funds
        reward_history.append(reward)
        
        # Update weights
        update_weights(model, reward, learning_rate)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {reward:.2f}")
            print(f"Current Portfolio Value: ${final_value:.2f}")
            print(f"Number of shares owned: {stock_owned}")
    
    # === Plot Results ===
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(reward_history, label='Episode Reward')
    plt.xlabel("Episode")
    plt.ylabel("Profit ($)")
    plt.title("LSM Trading Model - Training Progress")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.hist(reward_history, bins=20, label='Reward Distribution')
    plt.xlabel("Reward ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Trading Results")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final performance
    print(f"\nFinal Performance:")
    print(f"Average Reward: ${np.mean(reward_history):.2f}")
    print(f"Best Reward: ${np.max(reward_history):.2f}")
    print(f"Worst Reward: ${np.min(reward_history):.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Standard Deviation of Rewards: ${np.std(reward_history):.2f}")

if __name__ == "__main__":
    main() 