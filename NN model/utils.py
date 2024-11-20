# utils.py

# Mapping for label values to class indices and vice versa
label_map = {-1: 0, 0: 1, 1: 2}  # map -1 -> 0, 0 -> 1, 1 -> 2
inverse_label_map = {v: k for k, v in label_map.items()}

def map_label(label):
    return label_map[label]

def inverse_map_label(class_index):
    return inverse_label_map[class_index]
