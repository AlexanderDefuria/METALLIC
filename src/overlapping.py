import numpy as np


def volume_overlap(X, y) -> float:
    classes = np.unique(y)
    totalvoverlap = 0
    counter = 0
    r, c = np.shape(np.array(X))
    for v0 in range(len(classes)):
        for v1 in range(len(classes)):
            if classes[v0] != classes[v1] and classes[v0] < classes[v1]:
                counter = counter + 1
                voverlap = 1.0
                for i in range(c):
                    max0 = -100000.0
                    max1 = -100000.0
                    min0 = 100000.0
                    min1 = 100000.0

                    for j in range(r):
                        if y[j] == classes[v0]:
                            if max0 < X[j, i]:
                                max0 = X[j, i]

                            if min0 > X[j, i]:
                                min0 = X[j, i]

                        if y[j] == classes[v1]:
                            if max1 < X[j, i]:
                                max1 = X[j, i]

                            if min1 > X[j, i]:
                                min1 = X[j, i]
                    if np.maximum(max0, max1) - np.minimum(min0, min1) == 0:
                        voverlap = 0
                    else:
                        voverlap = voverlap * ((np.minimum(max0, max1) - np.maximum(min0, min1)) / (np.maximum(max0, max1) - np.minimum(min0, min1)))
                totalvoverlap = totalvoverlap + voverlap
    avgvoverlap = totalvoverlap / counter

    return avgvoverlap
