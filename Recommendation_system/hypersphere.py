import warnings

import numpy as np
from sklearn import metrics
warnings.filterwarnings("ignore")
def create_hypersphere(X,y):
    pool = list(range(X.shape[0]))
    size = X.shape
    np.random.shuffle(pool)
    dist = metrics.pairwise.euclidean_distances(X)

    hyperCentres = []
    while len(pool) > 0:
        hs = [pool[0]]  # Initialize hypersphere
        centre = X[pool[0]]  # and its centre
        hsClass = y[pool[0]]  # Class of this hypersphere
        pool.remove(pool[0])  # Remove the initial point from the pool
        mostDistantPair = None

        while True and len(pool) > 0:
            dist = np.sqrt(np.sum((X[pool] - centre) ** 2, axis=1))
            nn = pool[np.argmin(dist)]  # Nearest neighbour index
            if y[nn] != hsClass:  # If this point belongs to a different class
                break  # conclude the set of points in this sphere
            hs.append(nn)  # Otherwise add it to the sphere
            pool.remove(nn)  # and remove it from the pool

            centre = np.mean(X[hs], axis=0)

        hyperCentres.append(list(centre) + [int(hsClass)])

    return hyperCentres
