import pandas as pd
from math import sqrt

def complexity(req_file):
    data = req_file
    data1 = data
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    def get_neighbors(train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors
    
    # Test distance function
    data = data.values.tolist()
    final_column = data1[data1.columns[-1]]
    values = final_column.value_counts().keys().tolist()
    req_class = values[-1]
    # print(req_class)
    cm_list = []
    for i in data:
        easy = 0
        diff = 0
        final = i[len(i)-1]
        if req_class == final:
            neighbors = get_neighbors(data, i, 6)
            neighbors = neighbors[1:]
            for neighbor in neighbors:
                # print(neighbor)
                if neighbor[len(neighbor)-1]!=req_class:
                    easy = easy +1
                else:
                    diff = diff + 0
            
            cm = (easy+diff)/5.0
            

            if cm > 0.5:
                item_cm = 1
            else:
                item_cm = 0
            cm_list.append(item_cm)
        else:
            continue
    final_cm = sum(cm_list)/len(cm_list)
    # print(final_cm)
    return final_cm