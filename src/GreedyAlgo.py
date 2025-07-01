import numpy as np
from itertools import chain

class GreedyAlgo:
    def __init__(self, applied_liner_sum):
        self.OptimizeAlgoApplied = applied_liner_sum

    def greedy_sub_matrices(self, matrix, vertical_split, horizontal_split):
        if not isinstance(vertical_split, list) or not isinstance(horizontal_split, list):
            return "Invalid input"  # Check if vertical and Horizontal are lists

        vertical_slices = np.vsplit(matrix, vertical_split) # does the vertical Split

        sub_matrices = []
        for i in range(len(vertical_slices)): # for each vertical split do horizontal split
            horizontal_slices = np.hsplit(vertical_slices[i], horizontal_split) # does the horizontal split
            sub_matrices.append(horizontal_slices[i]) # adds de correct sub matrix to the return

        return sub_matrices

    def greedy_linear_applied(self, cost_matrices, num_of_matrices):
        total_mapping = []
        for k in range(num_of_matrices):
            num_blocks = 4

            rows = len(cost_matrices[k])

            base_size = rows // num_blocks
            remainder = rows % num_blocks

            split_points = []
            acc = 0
            for i in range(num_blocks - 1):
                acc += base_size + (1 if i < remainder else 0)
                split_points.append(acc)
            sub_matrices = self.greedy_sub_matrices(cost_matrices[k], split_points, split_points)

            mapping_row = []
            mapping_col = []
            for j in range(len(sub_matrices)):
                #print(sub_matrices[j].shape)
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
                    col_len = split_points[j - 1]
                    row_len = split_points[j - 1]
                    complete_row_mapping.append((mapping_row[j] + row_len))
                    complete_col_mapping.append((mapping_col[j] + col_len))
            total_mapping.append([list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))])
        return total_mapping

