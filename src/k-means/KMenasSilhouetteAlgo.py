import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.MatrixReader import MatrixReader
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class KMenasSilhouetteAlgo:
    def kmeans_silhouette(self, matrix):
        silhouette_avg = []
        K = range(2, round(len(matrix[0]) / 2))
        for num_cluster in K:
            kmeans = KMeans(n_clusters=num_cluster)
            kmeans.fit(matrix.T)
            cluster_labels = kmeans.labels_

            silhouette_avg.append(silhouette_score(matrix.T, cluster_labels))
        num_clusters = K[np.argmax(silhouette_avg)]

        cluster_matrices = []
        kmeans_row = KMeans(n_clusters=num_clusters)
        kmeans_col = KMeans(n_clusters=num_clusters)
        kmeans_row.fit(matrix)
        kmeans_col.fit(matrix.T)
        row_clusters = kmeans_row.labels_
        col_clusters = kmeans_col.labels_

        row_unique, row_count = np.unique(row_clusters, return_counts=True)
        col_unique, col_count = np.unique(col_clusters, return_counts=True)

        row_ids = list(zip(row_unique, row_count))
        col_ids = list(zip(col_unique, col_count))

        row_ids_sorted = sorted(row_ids, key=lambda x: x[1], reverse=True)
        col_ids_sorted = sorted(col_ids, key=lambda x: x[1], reverse=True)

        ids = list(zip(list(zip(*row_ids_sorted))[0], list(zip(*col_ids_sorted))[0]))
        i = 0
        for row_id, col_id in ids:
            if (row_ids_sorted[i][1] <= col_ids_sorted[i][1]):
                row_indices = np.where(row_clusters == row_id)[0]
                col_indices = np.where(col_clusters == col_id)[0]
                sub_matrix = matrix[row_indices, :][:, col_indices]
                cluster_matrices.append(sub_matrix)
            else:
                cluster_matrices.append('Not injective')
            i += 1
        plt.plot(K, silhouette_avg, 'bx-')
        plt.xticks(K)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        return cluster_matrices


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_silhouette = KMenasSilhouetteAlgo()

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(0)

    sub_matrices = kmeans_silhouette.kmeans_silhouette(cost_matrices[0])
    for sub_matrix in sub_matrices:
        matrix_reader.print_sub_matrix(sub_matrix)

