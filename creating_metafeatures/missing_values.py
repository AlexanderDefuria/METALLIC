import numpy as np
import math
def handlingMissingValues(X,y,data_clean):

    option = data_clean

    if(option == 1):
        rows,cols = X.shape
        meansArray = []
        class_label_mapper = {}
        index =0
        for i in np.unique(y):
            class_label_mapper[i] = index
            index+=1

        for i in range(len(np.unique(y))):
            meansArray.append([])
            for j in range(X.shape[1]):
                meansArray[i].append(np.nanmean(X[tuple(list(np.where(y == i)))][:, j]))
        for i in range(rows):
            for j in range(cols):
                if(math.isnan(X[i][j])):
                    if(math.isnan(meansArray[class_label_mapper[y[i]]][j])):
                        X[i][j] = 0
                    else:
                        X[i][j] = meansArray[class_label_mapper[y[i]]][j]
        return X

    elif(option == 2):
        X = X[~np.isnan(X).any(axis=1)]
        return X
    elif(option ==3):
        X = X[:,~np.isnan(X).any(axis=0)]
        return X
    else:
        print("Invalid data cleaning option\n")