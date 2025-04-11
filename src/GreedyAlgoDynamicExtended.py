import math
import numpy as np
from itertools import combinations
from itertools import chain
from src.OptimizeAlgoApplied import OptimizeAlgoApplied


class GreedyAlgoDynamicExtended():
    def __init__(self, OptimizeAlgoApplied):
        self.OptimizeAlgoApplied = OptimizeAlgoApplied

    def greedy_algo_extended(self, matrix, vertical_split, horizontal_split):
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
        mapping = self.OptimizeAlgoApplied.compute_linear_sum_assignment(horizontal_slices_sum)
        for i in range(len(mapping[0])):
            sub_matrices.append(horizontal_slices[mapping[0][i]][mapping[1][i]])
        return sub_matrices, mapping


    def greedy_linear_applied(self, cost_matrices, num_of_matrices, buckets_size):
        total_mapping = []
        for k in range(num_of_matrices):
            rows = len(cost_matrices[k])
            cols = len(cost_matrices[k][0])
            divisor = math.ceil(rows / buckets_size)
            row_interval = [round(rows * i / divisor) for i in range(1, divisor)]
            col_interval = [round(cols * i / divisor) for i in range(1, divisor)]
            sub_matrices, mapping = self.greedy_algo_extended(cost_matrices[k], row_interval, col_interval)

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
