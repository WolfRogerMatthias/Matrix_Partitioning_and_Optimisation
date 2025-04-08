import numpy as np
from src.MatrixReader import MatrixReader
from itertools import combinations
from itertools import chain
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
import h5py

class GreedyAlgo:
    def __init__(self, applied_liner_sum):
        self.OptimizeAlgoApplied = applied_liner_sum

    def greedy_sub_matrices(self, matrix, vertical_split, horizontal_split):
        if not isinstance(vertical_split, list) or not isinstance(horizontal_split, list):
            return "Invalid input"  # Check if vertical and Horizontal are lists

        vertical_slices = np.vsplit(matrix, vertical_split) # does the vertical Split

        sub_matrices = []
        for i in range(len(horizontal_split) + 1): # for each vertical split do horizontal split
            horizontal_slices = np.hsplit(vertical_slices[i], horizontal_split) # does the horizontal split
            sub_matrices.append(horizontal_slices[i]) # adds de correct sub matrix to the return

        return sub_matrices

    def greedy_linear_applied(self, cost_matrices, num_of_matrices):
        total_mapping = []
        for k in range(num_of_matrices):
            rows = len(cost_matrices[k])
            cols = len(cost_matrices[k][0])
            row_interval = [round(rows / 4), round(rows / 2), round(rows / 4) * 3]
            col_interval = [round(cols / 4), round(cols / 2), round(cols / 4) * 3]
            sub_matrices = self.greedy_sub_matrices(cost_matrices[k], row_interval, col_interval)

            mapping_row = []
            mapping_col = []
            for j in range(len(sub_matrices)):
                row, col = self.OptimizeAlgoApplied.compute_linear_sum_assignment(sub_matrices[j])
                mapping_row.append(row)
                mapping_col.append(col)

            complete_row_mapping = []
            complete_col_mapping = []
            for j in range(len(mapping_row)):
                if (j < 1):
                    complete_row_mapping.append(mapping_row[j])
                    complete_col_mapping.append(mapping_col[j])
                else:
                    col_len = col_interval[j - 1]
                    row_len = row_interval[j - 1]
                    complete_row_mapping.append((mapping_row[j] + row_len))
                    complete_col_mapping.append((mapping_col[j] + col_len))
            total_mapping.append([list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))])
        return total_mapping


if __name__ == '__main__':
    matrix_reader = MatrixReader()
    applied_liner_sum = OptimizeAlgoApplied()
    greedy_algo = GreedyAlgo(applied_liner_sum)

    num_of_matrix = 0

    len_dataset = 188

    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))
    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('./data/cost_matrices.h5', num_cost_matrices)

    total_mapping = greedy_algo.greedy_linear_applied(cost_matrices, num_cost_matrices)

    with h5py.File('./data/GreedyAlgo.h5', 'w') as file:
        for i in range(num_cost_matrices):
            file.create_dataset(f'matrix_mapping{i}', data=total_mapping[i])
        file.close()
    with h5py.File('./data/GreedyAlgo.h5', 'r') as file:
        test_mapping = {i: np.array(file[f'matrix_mapping{i}']) for i in range(num_cost_matrices)}
        file.close()

    matrix_reader.print_matrix(num_of_matrix)
    applied_liner_sum.print_linear_sum_assignment(test_mapping[num_of_matrix])
    matrix_reader.print_matrix(10000)
    applied_liner_sum.print_linear_sum_assignment(test_mapping[10000])