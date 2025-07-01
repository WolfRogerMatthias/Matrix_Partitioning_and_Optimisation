import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from KMeansAlgo import KMeansAlgo
from itertools import chain
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
import h5py


class KMenasSilhouetteAlgo:
    def __init__(self, kmeans_algo):
        self.kmeans_algo = kmeans_algo


    def kmeans_silhouette(self, matrix):
        silhouette_avg = []

        K = range(2, len(matrix) // 3)

        for num_cluster in K:
            kmeans = KMeans(n_clusters=num_cluster)
            kmeans.fit(matrix)
            cluster_labels = kmeans.labels_

            silhouette_avg.append(silhouette_score(matrix, cluster_labels))
        num_clusters = K[np.argmax(silhouette_avg)]

        cluster_matrices, row_indices_map, col_indices_map = self.kmeans_algo.kmeans_sub_matrix(matrix, num_clusters)
        return cluster_matrices, row_indices_map, col_indices_map
