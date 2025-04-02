import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.MatrixReader import MatrixReader
from src.optimize.OptimizeAlgoApplied import OptimizeAlgoApplied
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


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_algo = KMeansAlgo()
    applied_liner_sum = OptimizeAlgoApplied()

    num_matrix = 0

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('./src/data/cost_matrices.h5', num_cost_matrices)

    total_mapping = []

    for i in range(num_cost_matrices):
        sub_matrices, row_indices, col_indices = kmeans_algo.kmeans_sub_matrix(cost_matrices[i], 3)

        complete_row_mapping = []
        complete_col_mapping = []


        for j in range(len(sub_matrices)):
            row_mapping, col_mapping = applied_liner_sum.compute_linear_sum_assignment(sub_matrices[j])

            complete_row_mapping.append([row_indices[j][x] for x in row_mapping])
            complete_col_mapping.append([col_indices[j][x] for x in col_mapping])

        mapping = [list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))]
        total_mapping.append(mapping)

    with h5py.File('test.h5', 'w') as file:
        for i in range(num_cost_matrices):
            file.create_dataset(f'matrix_mapping{i}', data=total_mapping[i])
        file.close()
    with h5py.File('test.h5', 'r') as file:
        test_mapping = {i: np.array(file[f'matrix_mapping{i}']) for i in range(num_cost_matrices)}
        file.close()

    matrix_reader.print_matrix(num_matrix)
    applied_liner_sum.print_linear_sum_assignment(test_mapping[num_matrix])
    matrix_reader.print_matrix(10000)
    applied_liner_sum.print_linear_sum_assignment(test_mapping[10000])
    print(f'Number of completed mappings {len(kmeans_algo.number_of_turns) / num_cost_matrices * 100}%')

