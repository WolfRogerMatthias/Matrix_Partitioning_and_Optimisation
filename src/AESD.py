"""
This file contains the accuracy evaluation of the algorithms

The plots contain scattered plots of different matrix sizes minimal sum on them.
"""

# Imports
import time
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.MatrixDivider import MatrixDivider
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver
from src.AssignmentAlgo import AssignmentAlgo

# Init
MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
MatrixDivider = MatrixDivider()
ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()
AssignmentAlgo = AssignmentAlgo()

matrix_sizes = [i for i in range(100, 401, 20)]
number_of_matrices = 250

lsa_min_sums = []
divider_min_sums = []
bucket_min_sums = []
min_min_sums = []

print('Start')
for matrix_size in matrix_sizes:
    if matrix_size % 100 == 0:
        print(f'{matrix_size} run')
    cost_matrices = MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                                                   number_of_matrices)
    lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in
                   range(number_of_matrices)]
    divider_mapping = MatrixDivider.divider(cost_matrices, number_of_matrices, 4)
    bucket_mapping = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
    min_mapping = [AssignmentAlgo.assignment_applied(cost_matrices[i]) for i in range(number_of_matrices)]

    lsa_sum = []
    divider_sum = []
    bucket_sum = []
    min_sum = []

    for idx, key in enumerate(cost_matrices.keys()):
        matrix = cost_matrices[key]

        # LSA sum
        row_ind, col_ind = lsa_mapping[idx]
        lsa_sum.append(matrix[row_ind, col_ind].sum())

        # Divider sum
        divider_row = divider_mapping[idx][0]
        divider_col = divider_mapping[idx][1]
        divider_sum.append(matrix[divider_row, divider_col].sum())

        # Bucket sum
        bucket_pairs = bucket_mapping[idx]
        bucket_sum.append(sum(matrix[row, col] for row, col in bucket_pairs))

        # Min sum
        min_pairs = min_mapping[idx]
        min_sum.append(sum(matrix[row, col] for row, col in min_pairs.items()))


    lsa_min_sums.append(lsa_sum)
    divider_min_sums.append(divider_sum)
    bucket_min_sums.append(bucket_sum)
    min_min_sums.append(min_sum)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

lsa_arr = np.array(lsa_min_sums)
divider_arr = np.array(divider_min_sums)
bucket_arr = np.array(bucket_min_sums)
min_arr = np.array(min_min_sums)


# Color scheme
colors = {
    'baseline': '#333333',
    'algorithm1': '#FF8C00',  # Orange for Matrix Divider
    'algorithm2': '#800080',
    'algorithm3': '#0000FF'
}
"""

selected_indices = [0, 4, 8, 12]

# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, matrix_idx in enumerate(selected_indices):
    x_vals = np.array(lsa_min_sums[matrix_idx])
    y_vals = np.array(divider_min_sums[matrix_idx])

    axs[idx].scatter(x_vals, y_vals, color=colors['algorithm1'], alpha=0.7)

    # Diagonal reference line
    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())
    axs[idx].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    axs[idx].set_title(f'Matrix Size {matrix_sizes[matrix_idx]}x{matrix_sizes[matrix_idx]}')
    axs[idx].set_xlabel('LSA Min Sum')
    axs[idx].set_ylabel('Divider Min Sum')
    axs[idx].grid(True)

fig.suptitle('LSA vs Matrix Divider - Selected Matrix Sizes', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the suptitle

# Save plot to PNG
plt.savefig(f'./png/{timestamp}_scatter_lsa_matrix_divider.png')
plt.show()


selected_indices = [12, 16, 20, 24]


# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, matrix_idx in enumerate(selected_indices):
    x_vals = np.array(lsa_min_sums[matrix_idx])
    y_vals = np.array(min_min_sums[matrix_idx])

    axs[idx].scatter(x_vals, y_vals, color=colors['algorithm3'], alpha=0.7)

    # Diagonal reference line
    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())
    axs[idx].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    axs[idx].set_title(f'Matrix Size {matrix_sizes[matrix_idx]}x{matrix_sizes[matrix_idx]}')
    axs[idx].set_xlabel('LSA Min Sum')
    axs[idx].set_ylabel('Direct Min Assignment Min Sum')
    axs[idx].grid(True)

fig.suptitle('LSA vs Direct Min Assignment - Selected Matrix Sizes', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the suptitle

# Save plot to PNG
plt.savefig(f'./png/{timestamp}_scatter_lsa_direct_min.png')
plt.show()
"""


selected_indices = [4, 7, 10, 13]

# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, matrix_idx in enumerate(selected_indices):
    x_vals = np.array(lsa_min_sums[matrix_idx])
    y_vals = np.array(bucket_min_sums[matrix_idx])

    axs[idx].scatter(x_vals, y_vals, color=colors['algorithm2'], alpha=0.7)

    # Diagonal reference line
    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())
    axs[idx].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    axs[idx].set_title(f'Matrix Size {matrix_sizes[matrix_idx]}x{matrix_sizes[matrix_idx]}')
    axs[idx].set_xlabel('LSA Min Sum')
    axs[idx].set_ylabel('Bucket Assignment Min Sum')
    axs[idx].grid(True)

fig.suptitle('LSA vs Bucket Assignment - Selected Matrix Sizes', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the suptitle

# Save plot to PNG
plt.savefig(f'./png/{timestamp}_scatter_lsa_bucket.png')
plt.show()
