PreProcessData crunches csv HOLC into candlestick images and labels the data based on 
three regression lines and standard deviation. It normalized all so it can be used for 
diverse datasets/timeframes. 

# AI Trader Trainer

This repository contains various components for developing and testing AI-based trading strategies.

## Components

### 1. Data Processing
- **PreProcessData/main.py**: Processes CSV HOLC data into candlestick images
- Normalizes data for use across diverse datasets/timeframes

### 2. Image Filtering and Labeling
- **Labeling App/label_images_app.py**: Manual labeling application
- **Labeling App/review_labeled_images.py**: Verify label accuracy
- **Labeling App/update_all_labels.py**: Assign labels to all images
- Filtering based on linear regressions to ease the labeling process

### 3. Trading Simulation
- **trade_simulator.py**: Simulates basic buy/sell based on labeled data
- Uses labels and CSV data to evaluate trading performance

### 4. Feature Extraction
- **RESNET feature extractor**: ResNet18 model trained on labels across different timeframes
- Extracts meaningful features from candlestick images

### 5. Liquid State Machine (LSM) Implementation
- **LSM_experimets/Source/network.py**: LSM reservoir with log-normal weight distribution
- Processes market trend data using spiking neural networks
- Features include:
  - Biologically plausible log-normal weight distribution
  - Balanced input weights for positive and negative trends
  - Scalable architecture (now supporting 128 neurons)
  - Randomized input connectivity (30% of neurons)
  - Detailed visualization and statistics
- **LSM_experimets/Source/visualization.py**: Visualization tools for network activity and weights
- **LSM_experimets/Source/main.py**: Main script to run the LSM simulation
- See `LSM_experimets/README.md` for detailed documentation

## Workflow

1. Process CSV to images (PreProcessData/main.py)
2. Filter and label images (Labeling App/)
3. Review and verify labeled images
4. Run trading simulation to evaluate strategies
5. Apply neural networks (ResNet, LSM) to learn from labeled data
6. Use trained models to label new data and generate trading signals

## Future Work

- Integrate LSM reservoir with transformer models
- Develop end-to-end trading strategy using LSM outputs
- Implement reinforcement learning for optimizing trading parameters
- Expand to multi-timeframe analysis