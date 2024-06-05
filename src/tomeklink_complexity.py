import pandas as pd
import numpy as np
from math import sqrt

def tomelink_complexity(data):
    # Euclidean distance function
    def euclidean_distance(row1, row2):
        return sqrt(sum([(r1 - r2) ** 2 for r1, r2 in zip(row1, row2)]))

    # Function to get nearest neighbors
    def get_neighbors(train, test_row, num_neighbors):
        distances = [(train_row, euclidean_distance(test_row, train_row))
                     for train_row in train]
        distances.sort(key=lambda tup: tup[1])
        return [distances[i][0] for i in range(num_neighbors)]


    dataset = data.values.tolist()
    class_values = list(set(row[-1] for row in dataset))
    
    # Get the minority and majority classes
    class_counts = {cls: sum(row[-1] == cls for row in dataset) for cls in class_values}
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    # Calculate Tomek Links between the minority and majority classes
    tl_count = 0  # Tomek links count
    for row in dataset:
        if row[-1] == minority_class:
            # Get the nearest neighbor from the entire dataset
            neighbors = get_neighbors(dataset, row, 2)  # Get the nearest neighbor
            nearest_neighbor = neighbors[1]  # Exclude the instance itself
            
            # Check if the nearest neighbor is from the majority class and forms a Tomek link
            if nearest_neighbor[-1] == majority_class:
                # Check if the instance is the nearest neighbor's nearest neighbor
                reciprocal = get_neighbors(dataset, nearest_neighbor, 2)[1]
                if reciprocal == row:
                    tl_count += 1

    # Calculate TLCM for the minority and majority classes
    minority_instances = [row for row in dataset if row[-1] == minority_class]
    tlcm = float(tl_count) / len(minority_instances)  # TLCM calculation
    return tlcm