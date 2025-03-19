import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.MatrixReader import MatrixReader
from src.optimize.OptimizeAlgoApplied import OptimizeAlgoApplied
from itertools import chain


class KMeansAlgo:
    def kmeans_sub_matrix(self, matrix, num_clusters):
        counter = 0
        not_injective = False

        cluster_matrices = []
        row_indices_map = []
        col_indices_map = []

        while not not_injective and counter < 10:
            counter += 1
            not_injective = True
            cluster_matrices = []
            row_indices_map = []
            col_indices_map = []

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

            ids = list(zip(list(zip(*row_ids_sorted))[0],list(zip(*col_ids_sorted))[0]))
            i = 0
            for row_id, col_id in ids:
                if (row_ids_sorted[i][1] <= col_ids_sorted[i][1]):
                    row_indices = np.where(row_clusters == row_id)[0]
                    col_indices = np.where(col_clusters == col_id)[0]

                    sub_matrix = matrix[row_indices, :][:, col_indices]
                    cluster_matrices.append(sub_matrix)
                    row_indices_map.append(row_indices)
                    col_indices_map.append(col_indices)
                else:
                    not_injective = False
                i += 1
        return cluster_matrices, row_indices_map, col_indices_map


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_algo = KMeansAlgo()
    applied_liner_sum = OptimizeAlgoApplied()

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(0)

    sub_matrices, row_indices, col_indices = kmeans_algo.kmeans_sub_matrix(cost_matrices[0], 3)

    complete_row_mapping = []
    complete_col_mapping = []

    for i in range(len(sub_matrices)):
        matrix_reader.print_sub_matrix(sub_matrices[i])
        row_mapping, col_mapping = applied_liner_sum.compute_linear_sum_assignment(sub_matrices[i])
        applied_liner_sum.print_linear_sum_assignment_sub(i)
        complete_row_mapping.append([row_indices[i][x] for x in row_mapping])
        complete_col_mapping.append([col_indices[i][x] for x in col_mapping])

    total_mapping = [list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))]
    applied_liner_sum.print_linear_sum_assignment(total_mapping)