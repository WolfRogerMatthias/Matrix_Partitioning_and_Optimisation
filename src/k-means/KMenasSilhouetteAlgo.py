import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.MatrixReader import MatrixReader
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from KMeansAlgo import KMeansAlgo

class KMenasSilhouetteAlgo:
    def __init__(self, kmeans_algo):
        kmenas_algo = kmeans_algo


    def kmeans_silhouette(self, matrix):
        silhouette_avg = []
        K = range(2, round(len(matrix) / 2))
        for num_cluster in K:
            kmeans = KMeans(n_clusters=num_cluster)
            kmeans.fit(matrix)
            cluster_labels = kmeans.labels_

            silhouette_avg.append(silhouette_score(matrix, cluster_labels))
        num_clusters = K[np.argmax(silhouette_avg)]

        cluster_matrices = KMeansAlgo.kmeans_sub_matrix(matrix, num_clusters)

        plt.plot(K, silhouette_avg, 'bx-')
        plt.xticks(K)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        return cluster_matrices


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_algo = KMeansAlgo()
    kmeans_silhouette = KMenasSilhouetteAlgo(kmeans_algo)

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(0)

    sub_matrices = kmeans_silhouette.kmeans_silhouette(cost_matrices[0])
    for sub_matrix in sub_matrices:
        matrix_reader.print_sub_matrix(sub_matrix)

