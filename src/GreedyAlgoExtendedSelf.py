import numpy as np
from itertools import combinations
from itertools import chain
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.MockDataGenerator import MockDataGenerator
from src.GreedyAlgo import GreedyAlgo

class GreedyAlgoExtendedSelf:
    def __init__(self, OptimizeAlgoApplied):
        self.OptimizeAlgoApplied = OptimizeAlgoApplied

    def greedy_extended_sub_matrices(self, matrix, vertical_split, horizontal_split):
        if not (isinstance(vertical_split, list) and isinstance(horizontal_split, list)):
            return "Invalid input"  # Check if vertical and Horizontal are lists

        vertical_slices = np.vsplit(matrix, vertical_split)  # does the vertical Split

        sub_matrices = []
        horizontal_slices = []
        horizontal_slices_sum = []
        for i in range(len(horizontal_split) + 1):  # for each vertical split do horizontal split
            horizontal_slice = np.hsplit(vertical_slices[i], horizontal_split)  # does the horizontal split
            horizontal_slices.append(horizontal_slice)  # adds de correct sub matrix to the return
            horizontal_slices_sum.append([i.sum() for i in horizontal_slice])

        mapping_row = [i for i in range(len(horizontal_slices_sum))]
        mapping_col = []
        min_arg = np.argmin(horizontal_slices_sum, axis=1)
        for i in range(len(mapping_row)):
            while min_arg[i] not in mapping_col or i in mapping_row:
                if min_arg[i] not in mapping_col:
                    mapping_col.append(min_arg[i])
                    break
                else:
                    horizontal_slices_sum[i][min_arg[i]] = np.inf
                    min_arg[i] = np.argmin(horizontal_slices_sum[i])
        mapping = [mapping_row, mapping_col]
        for i in range(len(mapping[0])):
            sub_matrices.append(horizontal_slices[mapping[0][i]][mapping[1][i]])
        return sub_matrices, mapping

    def greedy_linear_applied(self, cost_matrices, num_of_matrices):
        total_mapping = []
        for k in range(num_of_matrices):
            rows, cols = cost_matrices[k].shape
            row_interval = [round(rows / 4), round(rows / 2), round(rows / 4) * 3]
            col_interval = [round(cols / 4), round(cols / 2), round(cols / 4) * 3]
            sub_matrices, mapping = self.greedy_extended_sub_matrices(cost_matrices[k], row_interval, col_interval)

            mapping_row = []
            mapping_col = []
            for j in range(len(sub_matrices)):
                row, col = self.OptimizeAlgoApplied.compute_linear_sum_assignment(sub_matrices[j])
                mapping_row.append(row)
                mapping_col.append(col)

            complete_row_mapping = []
            complete_col_mapping = []
            row_interval.insert(0, 0)
            col_interval.insert(0, 0)
            for j in range(len(mapping_row)):
                complete_row_mapping.append(mapping_row[j] + row_interval[mapping[0][j]])
                complete_col_mapping.append(mapping_col[j] + col_interval[mapping[1][j]])

            total_mapping.append([list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))])
        return total_mapping

if __name__ == '__main__':
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    GreedyAlgoExtendedSelf = GreedyAlgoExtendedSelf(OptimizeAlgoApplied)
    MockDataGenerator = MockDataGenerator()

    #MockDataGenerator.creating_mock_data('./data/mock_data_greedy.h5', 1, 20)
    matrix = MockDataGenerator.loadig_mock_data('./data/mock_data_greedy.h5', 1)

    test_mapping = GreedyAlgoExtendedSelf.greedy_linear_applied(matrix, 1)
    print()
    for i in range(len(test_mapping[0][0])):
        print(f'{test_mapping[0][0][i]} -> {test_mapping[0][1][i]}')






