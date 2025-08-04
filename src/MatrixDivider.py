from itertools import chain
import numpy as np
from scipy.optimize import linear_sum_assignment


class MatrixDivider:
    def __init__(self):
        self.lsa = 1
        
    def divider(self, matrix, number_of_matrices, n):
        total_mapping = []
        for k in range(number_of_matrices):
            rows, cols = matrix[k].shape

            # Calculate row split points
            base_row_size = rows // n
            remainder_row = rows % n
            row_sizes = np.full(n, base_row_size, dtype=int)
            row_sizes[:remainder_row] += 1
            # row_split_points = np.cumsum(row_sizes)[:-1] # Not strictly needed if iterating with start/end

            # Calculate column split points
            base_col_size = cols // n
            remainder_col = cols % n
            col_sizes = np.full(n, base_col_size, dtype=int)
            col_sizes[:remainder_col] += 1
            # col_split_points = np.cumsum(col_sizes)[:-1] # Not strictly needed if iterating with start/end

            row_mapping = []
            col_mapping = []

            row_start_current = 0
            col_start_current = 0  # Initialize outside the inner loop

            for idx in range(n):  # Iterate through diagonal blocks
                # Determine current row block boundaries
                row_size = row_sizes[idx]
                row_end_current = row_start_current + row_size

                # Determine current column block boundaries (same index for diagonal)
                col_size = col_sizes[idx]
                col_end_current = col_start_current + col_size

                # Extract the diagonal block using direct slicing (creates a view, no copy)
                block = matrix[k][
                    row_start_current:row_end_current,
                    col_start_current:col_end_current,
                ]

                # Perform Linear Sum Assignment
                row_ind, col_ind = linear_sum_assignment(block)

                # Adjust indices with offset
                row_mapping.append(row_ind + row_start_current)
                col_mapping.append(col_ind + col_start_current)

                # Update start points for the next iteration
                row_start_current = row_end_current
                col_start_current = col_end_current

            total_mapping.append(
                [np.concatenate(row_mapping).tolist(), np.concatenate(col_mapping).tolist()]
            )
        return total_mapping
