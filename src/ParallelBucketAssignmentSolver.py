import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


def solve_submatrix(args):
    sub_matrix, row_idx, col_offset, matrix_key = args
    row_ind, col_ind = linear_sum_assignment(sub_matrix)
    col_ind = col_ind + col_offset
    return [(matrix_key, row_idx[r], col_ind[c]) for r, c in zip(row_ind, range(len(col_ind)))]


class ParallelBucketAssignmentSolver:
    def solve_multiple(self, matrix_dict: Dict[str, np.ndarray], number_of_buckets: int) -> Dict[
        str, List[Tuple[int, int]]]:
        all_tasks = []

        for matrix_key, matrix in matrix_dict.items():
            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Expected np.ndarray for key '{matrix_key}', got {type(matrix)}")

            rows, cols = matrix.shape
            min_arg = np.argmin(matrix, axis=1)
            bucket_width = int(np.ceil(cols / number_of_buckets))

            # Initialize bucket stats as (max_col, min_col, count)
            bucket_stats = {i: (0, cols - 1, 0) for i in range(number_of_buckets)}

            # Initial fill of bucket stats based on min_arg
            for col_idx in min_arg:
                bucket_id = min(col_idx // bucket_width, number_of_buckets - 1)
                max_col, min_col, count = bucket_stats[bucket_id]
                max_col = max(max_col, col_idx)
                min_col = min(min_col, col_idx)
                count += 1
                bucket_stats[bucket_id] = (max_col, min_col, count)

            # Check if initial mapping fits in buckets
            correct_mapping = all(
                count <= (max_val - min_val + 1)
                for max_val, min_val, count in bucket_stats.values()
            )

            # Adjust bucket ranges until mapping fits
            while not correct_mapping:
                last_max = -1
                new_bucket_stats = {}

                for bucket_id in range(number_of_buckets):
                    if bucket_id == 0:
                        min_val = 0
                    else:
                        min_val = last_max + 1

                    if bucket_id == number_of_buckets - 1:
                        max_val = cols - 1
                    else:
                        # Extend enough to fit count rows
                        required_width = bucket_stats[bucket_id][2]
                        max_val = min_val + required_width - 1
                        # Prevent exceeding matrix width
                        if max_val >= cols:
                            max_val = cols - 1

                    new_bucket_stats[bucket_id] = (max_val, min_val, 0)
                    last_max = max_val

                # Reassign rows to buckets based on updated ranges
                for col_idx in min_arg:
                    for bucket_id, (max_val, min_val, count) in new_bucket_stats.items():
                        if min_val <= col_idx <= max_val:
                            max_val2, min_val2, count2 = new_bucket_stats[bucket_id]
                            new_bucket_stats[bucket_id] = (max_val2, min_val2, count2 + 1)
                            break

                bucket_stats = new_bucket_stats

                # Re-check if mapping fits in buckets
                correct_mapping = all(
                    count <= (max_val - min_val + 1)
                    for max_val, min_val, count in bucket_stats.values()
                )

            # Assign rows to buckets according to final bucket ranges
            assignments = {i: [] for i in range(number_of_buckets)}
            for row_idx, min_col in enumerate(min_arg):
                for bucket_id, (max_val, min_val, _) in bucket_stats.items():
                    if min_val <= min_col <= max_val:
                        assignments[bucket_id].append(row_idx)
                        break

            # Create submatrix tasks for parallel execution
            for bucket_id, row_indices in assignments.items():
                if not row_indices:
                    continue
                max_val, min_val, _ = bucket_stats[bucket_id]
                sub_matrix = matrix[np.array(row_indices), min_val:max_val + 1]
                all_tasks.append((sub_matrix, np.array(row_indices), min_val, matrix_key))

        # Run parallel solver on all tasks
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(solve_submatrix, all_tasks))

        # Collect and organize assignments by matrix key
        assignments_by_matrix: Dict[str, List[Tuple[int, int]]] = {key: [] for key in matrix_dict}
        for group in results:
            for matrix_key, row, col in group:
                assignments_by_matrix[matrix_key].append((row, col))

        # Sort assignments by row index for clarity
        for key in assignments_by_matrix:
            assignments_by_matrix[key].sort(key=lambda x: x[0])

        return assignments_by_matrix
