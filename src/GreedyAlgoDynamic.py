import math
import numpy as np
from itertools import combinations
from itertools import chain
from src.OptimizeAlgoApplied import OptimizeAlgoApplied

class GreedyAlgoDynamic():
    def __init__(self, OptimizeAlgoApplied):
        self.OptimizeAlgoApplied = OptimizeAlgoApplied

    def greedy_dynamic_sub_matrix(self, matrix, vertical_split, horizontal_split):
        if not isinstance(vertical_split, list) or not isinstance(horizontal_split, list):
            return "Invalid input"  # Check if vertical and Horizontal are lists

        vertical_slices = np.vsplit(matrix, vertical_split) # does the vertical Split

        sub_matrices = []
        for i in range(len(horizontal_split) + 1): # for each vertical split do horizontal split
            horizontal_slices = np.hsplit(vertical_slices[i], horizontal_split) # does the horizontal split
            sub_matrices.append(horizontal_slices[i]) # adds de correct sub matrix to the return

        return sub_matrices

    def greedy_linear_applied(self, cost_matrices, num_of_matrices, size):
        total_mapping = []
        for k in range(num_of_matrices):
            rows = len(cost_matrices[k])
            cols = len(cost_matrices[k][0])

            divisor = math.ceil(rows / size)
            row_interval = [round(rows * i / divisor) for i in range(1, divisor)]
            col_interval = [round(cols * i / divisor) for i in range(1, divisor)]
            sub_matrices = self.greedy_dynamic_sub_matrix(cost_matrices[k], row_interval, col_interval)

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



