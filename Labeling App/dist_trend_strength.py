import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_trend_strength_distribution(json_file, output_dir=None):
    """
    Plots the distribution of trend_strength from the JSON file.

    Parameters:
    - json_file: Path to the JSON file containing image metadata.
    - output_dir: Optional directory to save the plot.
    """
    # Load JSON data
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Extract trend_strength values
    trend_strengths = [
        attributes["trend_strength"]
        for attributes in json_data.values()
        if isinstance(attributes, dict)
    ]

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(trend_strengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Trend Strength", fontsize=16)
    plt.xlabel("Trend Strength", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', alpha=0.75)

    # Show the mean and standard deviation on the plot
    mean_trend = np.mean(trend_strengths)
    std_trend = np.std(trend_strengths)
    plt.axvline(mean_trend, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_trend:.2f}")
    plt.axvline(mean_trend + std_trend, color='green', linestyle='dashed', linewidth=1, label=f"Mean + 1 Std: {mean_trend + std_trend:.2f}")
    plt.axvline(mean_trend - std_trend, color='green', linestyle='dashed', linewidth=1, label=f"Mean - 1 Std: {mean_trend - std_trend:.2f}")
    plt.legend()

    # Save the plot to the output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "trend_strength_distribution.png")
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":   

    ticker = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe (e.g., '1d'): ")

    base_dir = os.path.join('data_processed_imgs', ticker, timeframe)
    json_file = os.path.join(base_dir, 'regression_data', f'{ticker}_{timeframe}_regression_data_normalized.json')

    # Plot the distribution
    plot_trend_strength_distribution(json_file)
