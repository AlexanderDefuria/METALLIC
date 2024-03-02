import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from itertools import product
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,fowlkes_mallows_score

def dbindex(X,labels):
    return davies_bouldin_score(X, labels)

def chindex(X,labels):
    return calinski_harabasz_score(X, labels)

def adjusted_rand(mlabels, labels):
    return adjusted_rand_score(mlabels, labels)

def adjusted_mutual_score(mlabels,labels):
    return adjusted_mutual_info_score(mlabels, labels)

def fowlkes_mallows(mlabels,labels):
    return fowlkes_mallows_score(mlabels, labels)

def normalized_mutual_info(mlabels, labels):
    return normalized_mutual_info_score(mlabels, labels)

def divide(data, labels):
    clusters = set(labels)
    clusters_data = []
    for cluster in clusters:
        clusters_data.append(data[labels == cluster, :])
    return clusters_data

def get_centroids(clusters):
    centroids = []
    for cluster_data in clusters:
        centroids.append(cluster_data.mean(axis=0))
    return centroids

def cohesion(data, labels):
    clusters = sorted(set(labels))
    sse = 0
    for cluster in clusters:
        cluster_data = data[labels == cluster, :]
        centroid = cluster_data.mean(axis = 0)
        sse += ((cluster_data - centroid)**2).sum()
    return sse

def separation(data, labels, cohesion_score):
    # calculate separation as SST - SSE
    return cohesion(data, np.zeros(data.shape[0])) - cohesion_score

def SST(data):
    c = get_centroids([data])
    return ((data - c) ** 2).sum()

def SSE(clusters, centroids):
    result = 0
    for cluster, centroid in zip(clusters, centroids):
        result += ((cluster - centroid) ** 2).sum()
    return result


# Clear the store before running each time
within_cluster_dist_sum_store = {}

def within_cluster_dist_sum(cluster, centroid, cluster_id):
    if cluster_id in within_cluster_dist_sum_store:
        return within_cluster_dist_sum_store[cluster_id]
    else:
        result = (((cluster - centroid) ** 2).sum(axis=1)**.5).sum()
        within_cluster_dist_sum_store[cluster_id] = result
    return result

def RMSSTD(data, clusters, centroids):
    df = data.shape[0] - len(clusters)
    attribute_num = data.shape[1]
    return (SSE(clusters, centroids) / (attribute_num * df)) ** .5

# equal to separation / (cohesion + separation)
def RS(data, clusters, centroids):
    sst = SST(data)
    sse = SSE(clusters, centroids)
    return (sst - sse) / sst

def DB_find_max_j(clusters, centroids, i):
    max_val = 0
    max_j = 0
    for j in range(len(clusters)):
        if j == i:
            continue
        cluster_i_stat = within_cluster_dist_sum(clusters[i], centroids[i], i) / clusters[i].shape[0]
        cluster_j_stat = within_cluster_dist_sum(clusters[j], centroids[j], j) / clusters[j].shape[0]
        val = (cluster_i_stat + cluster_j_stat) / (((centroids[i] - centroids[j]) ** 2).sum() ** .5)
        if val > max_val:
            max_val = val
            max_j = j
    return max_val

def DB(data, clusters, centroids):
    result = 0
    for i in range(len(clusters)):
        result += DB_find_max_j(clusters, centroids, i)
    return result / len(clusters)

def XB(data, clusters, centroids):
    sse = SSE(clusters, centroids)
    min_dist = ((centroids[0] - centroids[1]) ** 2).sum()
    for centroid_i, centroid_j in list(product(centroids, centroids)):
        if (centroid_i - centroid_j).sum() == 0:
            continue
        dist = ((centroid_i - centroid_j) ** 2).sum()
        if dist < min_dist:
            min_dist = dist
    return sse / (data.shape[0] * min_dist)


def get_validation_scores(data, labels):
    within_cluster_dist_sum_store.clear()

    clusters = divide(data, labels)
    centroids = get_centroids(clusters)

    scores = {}

    scores['cohesion'] = cohesion(data, labels)
    scores['separation'] = separation(data, labels, scores['cohesion'])
    #scores['calinski_harabaz_score'] = calinski_harabaz_score(data, labels)
    scores['RMSSTD'] = RMSSTD(data, clusters, centroids)
    scores['RS'] = RS(data, clusters, centroids)
    scores['DB'] = DB(data, clusters, centroids)
    scores['XB'] = XB(data, clusters, centroids)

    return scores
def k_Means(d,ts):
    no_of_cluster=len(np.unique(ts))
    if no_of_cluster ==1:
        no_of_cluster=2
    scores=[]
    kmeans = KMeans(init='k-means++', n_clusters=no_of_cluster)
    kmeans.fit(d)
    labels = kmeans.labels_
    score = metrics.silhouette_score(d, labels, metric='euclidean', sample_size=len(d))
    dbindex_score = dbindex(d, labels)
    chscore = chindex(d, labels)
    adjustrandscore = adjusted_rand(ts, labels)
    mutual_info_score = normalized_mutual_info_score(ts, labels)
    adjust_mutual = adjusted_mutual_score(ts, labels)
    fowlkes_score = fowlkes_mallows(ts, labels)
    within_cluster_dist_sum_store = {}
    all_score = get_validation_scores(d, labels)
    return score,dbindex_score,chscore,all_score['cohesion'],all_score['separation'],all_score['RMSSTD'],all_score['RS'],all_score['XB'],adjustrandscore,adjust_mutual,fowlkes_score,mutual_info_score