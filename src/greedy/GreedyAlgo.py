import numpy as np
from src.MatrixReader import MatrixReader
from itertools import combinations

class GreedyAlgo:

    def greedy_sub_matrices(self, matrix, vertical_split, horizontal_split):
        if not isinstance(vertical_split, list) or not isinstance(horizontal_split, list):
            return "Invalid input"  # Check if vertical and Horizontal are lists
        vertical_slices = np.vsplit(matrix, vertical_split)
        sub_matrices = []
        for i in range(len(horizontal_split) + 1):
            sub_matrices.append(np.hsplit(vertical_slices[i], horizontal_split)[i])
        return sub_matrices

if __name__ == "__main__":
    matrix_reader = MatrixReader()
    greedy_algo = GreedyAlgo()

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(10000)
    sub_matrices = []
    for i in range(num_cost_matrices):
        rows = len(cost_matrices[i])
        cols = len(cost_matrices[i][0])
        sub_matrices.append(greedy_algo.greedy_sub_matrices(cost_matrices[i], [round(rows / 4), round(rows / 2), round(rows / 4) * 3], [round(cols / 4), round(cols / 2), round(cols / 4) * 3]))

    for i in range(len(sub_matrices[10000])):
        matrix_reader.print_sub_matrix(sub_matrices[10000][i])

