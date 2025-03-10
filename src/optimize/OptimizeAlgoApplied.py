import numpy
from src.MatrixReader import MatrixReader
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from src.kmeans.KMeansAlgo import KMeansAlgo

class OptimizeAlgoApplied:
    def __init__(self, row_ind, col_ind):
        row_ind, col_ind = row_ind, col_ind


    def optimize_applied(self, matrix):
        if not isinstance(matrix, numpy.ndarray):
            return "Not injective", "Not injective"
        row, col = linear_sum_assignment(matrix)
        row_ind.append(row)
        col_ind.append(col)



if __name__ == '__main__':
    matrix_reader = MatrixReader()
    kmeans_algo = KMeansAlgo()
    row_ind, col_ind = [], []
    optimize_applied = OptimizeAlgoApplied(row_ind, col_ind)

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(0)

    sub_matrices = kmeans_algo.kmeans_sub_matrix(cost_matrices[0], 3)

    for i in range(len(sub_matrices)):
        matrix_reader.print_sub_matrix(sub_matrices[i])
        optimize_applied.optimize_applied(sub_matrices[i])

    for i in range(len(row_ind)):
        if not isinstance(row_ind[i], numpy.ndarray):
            print("Not injective")
        else:
            for j in range(len(row_ind[i])):
                print(f'{row_ind[i][j]} -> {col_ind[i][j]}')
        print('\n')

