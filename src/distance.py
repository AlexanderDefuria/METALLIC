import numpy as np


def distance(hypersphere, different_classes):
    final_distance = []
    for i in different_classes:
        for j in different_classes:
            if (i != j) and (i < j):
                get_all_i = np.array(hypersphere[hypersphere[:, hypersphere.shape[1] - 1] == i])
                get_all_j = np.array(hypersphere[hypersphere[:, hypersphere.shape[1] - 1] == j])
                total_distance = []
                for q in get_all_i:
                    distance_merge = []
                    for k in get_all_j:
                        distance = np.linalg.norm(q - k)
                        distance_merge.append(distance)
                    total_distance.append(min(distance_merge))
                final_distance.append(min(total_distance))
    avg_distance = sum(final_distance) / len(final_distance)
    return avg_distance
