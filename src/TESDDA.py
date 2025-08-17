"""
This fiel is for the timing Evaluation for the direct min Assignment algo
"""

import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.AssignmentAlgo import AssignmentAlgo

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
AssignmentAlgo = AssignmentAlgo()

matrix_sizes = [i for i in range(50, 151, 5)]
number_of_matrices = 1000
number_of_runs = 10

lsa_timings = []
direct_timings = []
for i in range(number_of_runs):
    print(f'start of run {i + 1}', end='')
    start = time.time()
    lsa_run = []
    direct_run = []
    for matrix_size in matrix_sizes:
        cost_matrices = MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                                                       number_of_matrices)
        lsa_start = time.time()
        lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[j]) for j in range(number_of_matrices)]
        lsa_end = time.time()

        direct_start = time.time()
        direct_mapping = [AssignmentAlgo.assignment_applied(cost_matrices[j]) for j in range(number_of_matrices)]
        direct_end = time.time()

        lsa_run.append(lsa_end - lsa_start)
        direct_run.append(direct_end - direct_start)

    lsa_timings.append(lsa_run)
    direct_timings.append(direct_run)
    end = time.time()
    print(f'end of run, time elapsed: {end - start: 6.2f} seconds')

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
colors = {
    'lsa': '#333333',  # Black
    'divider': '#FF8C00',  # Orange
    'bucket': '#800080',  # Purple
    'direct': '#0000FF'  # Blue
}

lsa_arr = np.array(lsa_timings)
direct_arr = np.array(direct_timings)

lsa_mean = np.mean(lsa_arr, axis=0)
direct_mean = np.mean(direct_arr, axis=0)

def trimmed_min_max(arr):
    """Return min and max after removing the smallest and largest value per column."""
    trimmed_min = []
    trimmed_max = []
    for col in arr.T:  # iterate over each matrix size's results
        sorted_col = np.sort(col)
        trimmed = sorted_col[1:-1]  # remove first and last (min and max)
        trimmed_min.append(np.min(trimmed))
        trimmed_max.append(np.max(trimmed))
    return np.array(trimmed_min), np.array(trimmed_max)


lsa_min, lsa_max = trimmed_min_max(lsa_arr)
direct_min, direct_max = trimmed_min_max(direct_arr)


# First plot: timings
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, lsa_mean, label='LSA (Average)', color=colors['lsa'], linewidth=2)
plt.fill_between(matrix_sizes, lsa_min, lsa_max, color=colors['lsa'], alpha=0.2, label='LSA Range')

plt.plot(matrix_sizes, direct_mean, label='Direct Min (Average)', color=colors['direct'], linewidth=2)
plt.fill_between(matrix_sizes, direct_min, direct_max, color=colors['direct'], alpha=0.2,
                 label='Direct Min Range')
plt.legend()
plt.xticks(matrix_sizes)
plt.xlabel("Matrix Size (n x n)")
plt.ylabel("Execution Time (s)")
plt.title("LSA vs Direct Min - Timing Comparison")

plt.grid(True)
plt.ylim(0, None)
plt.xlim(matrix_sizes[0], matrix_sizes[-1])

plt.tight_layout()
plt.savefig(f'./png/{timestamp}_lsa_vs_direct_timing.png')
plt.show()
