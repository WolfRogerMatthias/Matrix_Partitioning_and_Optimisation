import numpy as np

class AssignmentAlgo:
    def __init__(self):
        self.sub_matrices = []

    def assignment_applied(self, matrix):
        n_row, n_col = matrix.shape

        available_cols = np.ones(n_col, dtype=bool)
        assignment = {}

        for row in range(n_row):
            row_values = matrix[row]
            masked_row = np.where(available_cols, row_values, np.inf)

            min_col = np.argmin(masked_row)

            if not np.isinf(masked_row[min_col]):
                assignment[row] = min_col
                available_cols[min_col] = False
        return assignment
