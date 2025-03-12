import nest
import numpy as np
import json
import matplotlib.pyplot as plt

# === Load Market Data ===
with open("market_data.json", "r") as file:
    market_data = json.load(file)  # Dictionary with {'date': {'price': X, 'trend': Y}}

# Convert JSON into structured lists
dates = list(market_data.keys())
prices = np.array([market_data[date]['price'] for date in dates])
trends = np.array([market_data[date]['trend'] for date in dates])

# === Simulation Parameters ===
initial_funds = 10_000  # Starting capital
fixed_trade_size = 100   # Amount of stock bought/sold per trade
gamma = 0.99             # Discount factor
learning_rate = 0.01     # RL learning rate
num_episodes = 50        # Training iterations
sim_time = 50.0          # Simulation time per step (ms)

# === Setup NEST ===
nest.ResetKernel()

# Create input neurons (2 inputs: trend > 100 and trend < -100)
input_neurons = nest.Create("parrot_neuron", 2)  # Parrot neurons forward spikes

# Create LSM Reservoir
num_reservoir_neurons = 20
reservoir = nest.Create("iaf_psc_alpha", num_reservoir_neurons)

# Create output neurons (2 outputs: buy and sell)
output_neurons = nest.Create("iaf_psc_alpha", 2)

# Connect inputs to reservoir
nest.Connect(input_neurons, reservoir, {"rule": "pairwise_bernoulli", "p": 0.3})

# Connect reservoir to output neurons
nest.Connect(reservoir, output_neurons, {"rule": "pairwise_bernoulli", "p": 0.3})

# Create synapses with trainable weights
synapses = nest.Create("static_synapse", len(reservoir), params={"weight": np.random.rand(len(reservoir))})

# Connect the synapses
nest.Connect(reservoir, output_neurons, synapses)

# Create a spike detector
spike_detector = nest.Create("spike_detector")
nest.Connect(output_neurons, spike_detector)

# === Reinforcement Learning ===
balance = initial_funds
stock_owned = 0
reward_history = []

for episode in range(num_episodes):
    balance = initial_funds
    stock_owned = 0
    
    for i in range(len(prices)):
        # === Convert Trend to Spikes ===
        if trends[i] > 100:
            nest.SpikeGenerator(input_neurons[0], [sim_time])  # Generate a spike
        elif trends[i] < -100:
            nest.SpikeGenerator(input_neurons[1], [sim_time])  # Generate a spike

        # === Simulate LSM ===
        nest.Simulate(sim_time)

        # === Get Output Spikes ===
        spike_counts = nest.GetStatus(spike_detector, keys="n_events")[0]

        # === Execute Trades Based on Spikes ===
        if spike_counts[0] > 0:  # Buy signal
            if balance >= fixed_trade_size * prices[i]:
                balance -= fixed_trade_size * prices[i]
                stock_owned += fixed_trade_size

        if spike_counts[1] > 0:  # Sell signal
            if stock_owned >= fixed_trade_size:
                balance += fixed_trade_size * prices[i]
                stock_owned -= fixed_trade_size

    # Final value of holdings
    final_value = balance + (stock_owned * prices[-1])

    # Compute reward as profit
    reward = final_value - initial_funds
    reward_history.append(reward)

    # === RL Policy Update (REINFORCE) ===
    for j in range(len(synapses)):
        action = np.random.choice([0, 1], p=[0.5, 0.5])  # Random action for training
        synaptic_weights = nest.GetStatus(synapses, keys="weight")
        new_weight = synaptic_weights[j] + learning_rate * (reward - np.mean(reward_history)) * action
        new_weight = np.clip(new_weight, 0, 1)  # Keep weights stable
        nest.SetStatus([synapses[j]], {"weight": new_weight})

# === Plot Results ===
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Profit ($)")
plt.title("LSM RL Training - Trading Profit Over Time")
plt.show()
