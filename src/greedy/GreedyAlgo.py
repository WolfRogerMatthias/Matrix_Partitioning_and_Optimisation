import numpy as np
from src.MatrixReader import MatrixReader
from itertools import combinations
from itertools import chain
from src.optimize.OptimizeAlgoApplied import OptimizeAlgoApplied

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
    applied_liner_sum = OptimizeAlgoApplied()

    num_of_matrix = 0

    len_dataset = 188
    num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

    print(f'Number of cost matrices: {num_cost_matrices}')

    cost_matrices = matrix_reader.load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

    matrix_reader.print_matrix(num_of_matrix)
    sub_matrices = []
    row_intervals = []
    col_intervals = []
    for i in range(num_cost_matrices):
        rows = len(cost_matrices[i])
        cols = len(cost_matrices[i][0])
        row_interval = [round(rows / 4), round(rows / 2), round(rows / 4) * 3]
        col_interval = [round(cols / 4), round(cols / 2), round(cols / 4) * 3]
        row_intervals.append(row_interval)
        col_intervals.append(col_interval)
        sub_matrices.append(greedy_algo.greedy_sub_matrices(cost_matrices[i], row_interval, col_interval))

    for i in range(len(sub_matrices[num_of_matrix])):
        matrix_reader.print_sub_matrix(sub_matrices[num_of_matrix][i])

    mapping_row = []
    mapping_col = []
    for i in range(len(sub_matrices[num_of_matrix])):
        row, col = applied_liner_sum.compute_linear_sum_assignment(sub_matrices[num_of_matrix][i])
        mapping_row.append(row)
        mapping_col.append(col)

    for i in range(len(mapping_row)):
        applied_liner_sum.print_linear_sum_assignment_sub(i)

    complete_row_mapping = []
    complete_col_mapping = []
    for i in range(len(mapping_row)):
        if (i < 1):
            complete_row_mapping.append(mapping_row[i])
            complete_col_mapping.append(mapping_col[i])
        else:
            col_len = col_intervals[num_of_matrix][i - 1]
            row_len = row_intervals[num_of_matrix][i - 1]
            complete_row_mapping.append((mapping_row[i] + row_len))
            complete_col_mapping.append((mapping_col[i] + col_len))


    total_mapping = [list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))]
    applied_liner_sum.print_linear_sum_assignment(total_mapping)

    test_row, test_col = applied_liner_sum.compute_linear_sum_assignment(cost_matrices[num_of_matrix])
    test_map = [test_row, test_col]
    applied_liner_sum.print_linear_sum_assignment(test_map)

