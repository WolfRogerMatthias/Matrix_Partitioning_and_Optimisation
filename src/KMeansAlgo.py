import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from itertools import chain
import h5py


class KMeansAlgo:
    def __init__(self):
        self.number_of_turns = []

    def kmeans_sub_matrix(self, matrix, num_clusters):
        counter = 0
        not_injective = True

        cluster_matrices = []
        row_indices_map = []
        col_indices_map = []
        while not_injective and counter < 20:
            counter += 1
            not_injective = False
            cluster_matrices.clear()
            row_indices_map.clear()
            col_indices_map.clear()

            kmeans_row = KMeans(n_clusters=num_clusters)
            kmeans_col = KMeans(n_clusters=num_clusters)

            kmeans_row.fit(matrix)
            kmeans_col.fit(matrix.T)

            row_clusters = kmeans_row.labels_
            col_clusters = kmeans_col.labels_

            row_count = np.bincount(row_clusters)
            col_count = np.bincount(col_clusters)

            row_sorted_ids = np.argsort(row_count)[::-1]
            col_sorted_ids = np.argsort(col_count)[::-1]

            ids = list(zip(row_sorted_ids, col_sorted_ids))
            i = 0
            for row_id, col_id in ids:
                if row_count[row_id] <= col_count[col_id]:
                    row_indices = np.where(row_clusters == row_id)[0]
                    col_indices = np.where(col_clusters == col_id)[0]

                    sub_matrix = matrix[row_indices, :][:, col_indices]
                    cluster_matrices.append(sub_matrix)
                    row_indices_map.append(row_indices)
                    col_indices_map.append(col_indices)
                else:
                    not_injective = True
                i += 1
            if not not_injective:
                self.number_of_turns.append(counter)
        return cluster_matrices, row_indices_map, col_indices_map
