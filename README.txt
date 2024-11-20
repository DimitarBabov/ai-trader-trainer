1. process csv to images
PreProcessData/main.py

2. Filter Images and Label them
Filtering images based on linear regressions to ease the labeling process.
Label images manually using app
Labeling App/ label_images_app.py

4. review labeled images
Labeling App/ review_labeled_images.py - make sure there is no error in the labels
Labeling App/ update_all_labels.py - assign labels to all images

5. Trading Simulator
looks at the labels and data csv file and simulates basic buy and sell if labels indicate
trade_simulator.py

6.Neural network that learns from labels and applies to new data to be labeled
to do..