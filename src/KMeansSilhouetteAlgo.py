import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from src.MatrixReader import MatrixReader
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


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_algo = KMeansAlgo()
    kmeans_silhouette = KMenasSilhouetteAlgo(kmeans_algo)
    applied_liner_sum = OptimizeAlgoApplied()

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('./data/cost_matrices.h5', num_cost_matrices)

    total_mapping = []
    for i in range(num_cost_matrices):
        sub_matrices, row_indices, col_indices = kmeans_silhouette.kmeans_silhouette(cost_matrices[i])

        complete_row_mapping = []
        complete_col_mapping = []


        for j in range(len(sub_matrices)):
            row_mapping, col_mapping = applied_liner_sum.compute_linear_sum_assignment(sub_matrices[j])

            complete_row_mapping.append([row_indices[j][x] for x in row_mapping])
            complete_col_mapping.append([col_indices[j][x] for x in col_mapping])

        mapping = [list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))]
        total_mapping.append(mapping)


    with h5py.File('./data/KmeansAlgoSilhouetteMapping.h5', 'w') as file:
        for i in range(num_cost_matrices):
            file.create_dataset(f'matrix_mapping{i}', data=total_mapping[i])
        file.close()
    with h5py.File('./data/KmeansAlgoSilhouetteMapping.h5', 'r') as file:
        test_mapping = {i: np.array(file[f'matrix_mapping{i}']) for i in range(num_cost_matrices)}
        file.close()

    matrix_reader.print_matrix(0)
    applied_liner_sum.print_linear_sum_assignment(test_mapping[0])
    matrix_reader.print_matrix(10000)
    applied_liner_sum.print_linear_sum_assignment(test_mapping[10000])
    print(f'Number of completed mappings {len(kmeans_algo.number_of_turns) / num_cost_matrices * 100}%')
