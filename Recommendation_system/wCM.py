import pandas as pd
from math import sqrt
def weighted_complexity(req_file):
    data = req_file
    data1=data
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
            res_list = []
            val_list= []
            neighbors = get_neighbors(data, i, 6)
            neighbors = neighbors[1:]
            distances=[]
            d1= euclidean_distance(i, neighbors[0])
            d2= euclidean_distance(i, neighbors[1])
            d3= euclidean_distance(i, neighbors[2])
            d4= euclidean_distance(i, neighbors[3])
            d5= euclidean_distance(i, neighbors[4])
            w1=1
            w5=0
            try:
                w2=(d5-d2)/(d5-d1)
            except:
                w2=0
            try:
                w3=(d5-d3)/(d5-d1)
            except:
                w3=0
            try:
                w4=(d5-d4)/(d5-d1)
            except:
                w4=0
            distances.append(w1)
            distances.append(w2)
            distances.append(w3)
            distances.append(w4)
            distances.append(w5)
            for neighbor in neighbors:
                # print(neighbor)
                if neighbor[len(neighbor)-1]!=req_class:
                    val_list.append(1)
                else:
                    val_list.append(0)

            for i in range(0, len(val_list)):
                res_list.append(val_list[i] * distances[i])
            cm = sum(res_list)/sum(distances)
            

            if cm > 0.5:
                item_cm = 1
            else:
                item_cm = 0
            cm_list.append(item_cm)
        else:
            continue
    final_cm = sum(cm_list)/len(cm_list)
    return final_cm
