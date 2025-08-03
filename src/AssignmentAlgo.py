import numpy as np
from scipy.optimize import linear_sum_assignment

class AssignmentAlgo:
    def __init__(self):
        self.sub_matrices = []

    def assignment_applied(self, matrix):
        n_rows, n_cols = matrix.shape
        assignment = {}
        available_col_set = set()
        other_rows = []

        for row in range(n_rows):
            row_values = matrix[row]
            min_col = np.argmin(row_values)

            if min_col not in available_col_set:
                available_col_set.add(min_col)
                assignment[row] = min_col
            else:
                other_rows.append(row)
        all_cols = set(range(n_cols))
        unassigned_cols = list(all_cols - available_col_set)
        sub_matrix = np.array([])
        if other_rows and unassigned_cols:
            sub_matrix = matrix[np.ix_(other_rows, unassigned_cols)]
            row_ind, col_ind = linear_sum_assignment(sub_matrix)

            # Map submatrix results back to original indices
            for i in range(len(row_ind)):
                original_row = other_rows[row_ind[i]]
                original_col = unassigned_cols[col_ind[i]]
                assignment[original_row] = original_col
        sorted_assignment = dict(sorted(assignment.items()))
        return sorted_assignment
