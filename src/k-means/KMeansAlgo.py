import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.MatrixReader import MatrixReader


class KMeansAlgo:
    def kmeans_sub_matricx(self, matrix, num_clusters):
        cluster_matrices = []
        kmeans_row = KMeans(n_clusters=num_clusters)
        kmeans_col = KMeans(n_clusters=num_clusters)
        kmeans_row.fit(matrix)
        kmeans_col.fit(matrix.T)
        row_clusters = kmeans_row.labels_
        col_clusters = kmeans_col.labels_
        # TODO: Map biggest to biggest cluster if no col cluster bigger then row cluster throw error because no iniectv function
        ids = list(zip(np.arange(num_clusters), np.arange(num_clusters)))
        for row_id, col_id in ids:
            row_indices = np.where(row_clusters == row_id)[0]
            col_indices = np.where(col_clusters == col_id)[0]

            sub_matrix = matrix[row_indices, :][: , col_indices]
            cluster_matrices.append(sub_matrix)
        return cluster_matrices


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_algo = KMeansAlgo()

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(0)
    sub_matrices = kmeans_algo.kmeans_sub_matricx(cost_matrices[0], 4)
    matrix_reader.print_sub_matrix(sub_matrices[0])
    matrix_reader.print_sub_matrix(sub_matrices[1])
    matrix_reader.print_sub_matrix(sub_matrices[2])
    matrix_reader.print_sub_matrix(sub_matrices[3])
