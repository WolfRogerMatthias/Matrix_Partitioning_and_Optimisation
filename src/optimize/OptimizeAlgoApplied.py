from scipy.optimize import linear_sum_assignment


class OptimizeAlgoApplied:
    def __init__(self):
        self.row_ind, self.col_ind = [], []

    def compute_linear_sum_assignment(self, matrix):
        row, col = linear_sum_assignment(matrix)
        self.row_ind.append(row)
        self.col_ind.append(col)
        return row, col

    def print_linear_sum_assignment_sub(self, num_mapping):
        for i in range(len(self.row_ind[num_mapping])):
            print(f'{self.row_ind[num_mapping][i]} -> {self.col_ind[num_mapping][i]}')
        print('')

    def print_linear_sum_assignment(self, mapping):
        for i in range(len(mapping[0])):
            print(f'{mapping[0][i]} -> {mapping[1][i]}')
        print('')

