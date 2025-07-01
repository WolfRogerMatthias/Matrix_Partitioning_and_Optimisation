from itertools import chain
import numpy as np
from scipy.optimize import linear_sum_assignment


class MatrixDivider:
    def __init__(self):
        self.lsa = 1
        
    def divider(self, matrix, number_of_matrices, n):
        total_mapping = []
        for k in range(number_of_matrices):
            rows, cols = matrix[k].shape  # Get rows and columns separately

            # Row splits
            base_row_size = rows // n
            remainder_row = rows % n
            row_sizes = np.full(n, base_row_size, dtype=int)
            row_sizes[:remainder_row] += 1
            row_split_points = np.cumsum(row_sizes)[:-1]

            # Column splits (independent of row)
            base_col_size = cols // n
            remainder_col = cols % n
            col_sizes = np.full(n, base_col_size, dtype=int)
            col_sizes[:remainder_col] += 1
            col_split_points = np.cumsum(col_sizes)[:-1]

            row_mapping = []
            col_mapping = []

            row_start = 0
            for idx, row_size in enumerate(row_sizes):
                row_end = row_start + row_size

                col_start = 0
                for jdx, col_size in enumerate(col_sizes):
                    col_end = col_start + col_size

                    if idx == jdx:  # Only process diagonal blocks
                        block = matrix[k][row_start:row_end, col_start:col_end]
                        #print(f"Block shape: {block.shape}")
                        row, col = linear_sum_assignment(block)

                        # Compute row/col offset correctly from split points
                        if idx == 0:
                            row_mapping.append(row)
                            col_mapping.append(col)
                        else:
                            row_offset = row_split_points[idx - 1]
                            col_offset = col_split_points[jdx - 1]
                            row_mapping.append(row + row_offset)
                            col_mapping.append(col + col_offset)

                    col_start = col_end
                row_start = row_end

            total_mapping.append([
                np.concatenate(row_mapping).tolist(),
                np.concatenate(col_mapping).tolist()
            ])
        return total_mapping
