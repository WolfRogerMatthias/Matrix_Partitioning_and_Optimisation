"""
This file is for the scientific data timing evaluation of the 3 algorithms in my Thesis.
It will generate 4 Plots that are the following
1. LSA vs Matrix Divider
2. LSA vs Bucket
3. LSA vs Min Assignment
4. LSA vs all the others
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
number_of_runs = 10

lsa_timings = []
divider_timings = []
bucket_timings = []
min_timings = []

total_time = time.time()
for i in range(number_of_runs):
    run_start = time.time()
    print(f"Run {i + 1} start", end='')
    lsa_timings_run = []
    divider_timings_run = []
    bucket_timings_run = []
    min_timings_run = []
    for matrix_size in matrix_sizes:
        cost_matrices = MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                                                       number_of_matrices)
        lsa_start = time.time()
        lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in
                       range(number_of_matrices)]
        lsa_end = time.time()

        divider_start = time.time()
        # divider_mapping = MatrixDivider.divider(cost_matrices, number_of_matrices, 4)
        divider_end = time.time()

        bucket_start = time.time()
        bucket_mapping = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
        bucket_end = time.time()

        min_start = time.time()
        # min_mapping = [AssignmentAlgo.assignment_applied(cost_matrices[i]) for i in range(number_of_matrices)]
        min_end = time.time()

        lsa_timings_run.append(lsa_end - lsa_start)
        divider_timings_run.append(divider_end - divider_start)
        bucket_timings_run.append(bucket_end - bucket_start)
        min_timings_run.append(min_end - min_start)
    lsa_timings.append(lsa_timings_run)
    divider_timings.append(divider_timings_run)
    bucket_timings.append(bucket_timings_run)
    min_timings.append(min_timings_run)
    run_end = time.time()
    print(f' end of run time elapsed: {run_end - run_start:6.2f} seconds')

# Plot init

timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

colors = {
    'baseline': '#333333',  # Black
    'algorithm1': '#FF8C00',  # Orange
    'algorithm2': '#800080',  # Purple
    'algorithm3': '#0000FF'  # Blue
}

lsa_arr = np.array(lsa_timings)
divider_arr = np.array(divider_timings)
bucket_arr = np.array(bucket_timings)
min_arr = np.array(min_timings)

lsa_mean = np.mean(lsa_arr, axis=0)
lsa_min = np.min(lsa_arr, axis=0)
lsa_max = np.max(lsa_arr, axis=0)

divider_mean = np.mean(divider_arr, axis=0)
divider_min = np.min(divider_arr, axis=0)
divider_max = np.max(divider_arr, axis=0)

bucket_mean = np.mean(bucket_arr, axis=0)
bucket_min = np.min(bucket_arr, axis=0)
bucket_max = np.max(bucket_arr, axis=0)

min_mean = np.mean(min_arr, axis=0)
min_min = np.min(min_arr, axis=0)
min_max = np.max(min_arr, axis=0)
"""
# Plot 1
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, lsa_mean, label='LSA (Average)', color=colors['baseline'], linewidth=2)
plt.fill_between(matrix_sizes, lsa_min, lsa_max, color=colors['baseline'], alpha=0.2, label='LSA Range')

plt.plot(matrix_sizes, divider_mean, label='Matrix Divider (Average)', color=colors['algorithm1'], linewidth=2)
plt.fill_between(matrix_sizes, divider_min, divider_max, color=colors['algorithm1'], alpha=0.2,
                 label='Matrix Divider Range')

plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Time (seconds)')
plt.title('LSA vs Matrix Divider - Runtime Comparison')
plt.legend()
plt.xticks([i for i in range(10, 81, 5)])
plt.ylim(0, lsa_timings[0][15])
plt.xlim(10, 81)
plt.grid(True)
plt.tight_layout()

plt.savefig(f'./png/{timestamp}_lsa_vs_matrix_divider_timing.png')
plt.show()
"""
# Plot 2

plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, lsa_mean, label='LSA (Average)', color=colors['baseline'], linewidth=2)
plt.fill_between(matrix_sizes, lsa_min, lsa_max, color=colors['baseline'], alpha=0.2, label='LSA Range')

plt.plot(matrix_sizes, bucket_mean, label='Bucket Mapping (Average)', color=colors['algorithm2'], linewidth=2)
plt.fill_between(matrix_sizes, bucket_min, bucket_max, color=colors['algorithm2'], alpha=0.2,
                 label='Bucket Mapping Range')

plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Time (seconds)')
plt.title('LSA vs Bucket Mapping - Runtime Comparison')
plt.legend()
plt.xticks([i for i in range(100, 401, 20)])
plt.ylim(0, None)
plt.xlim(100, 401)
plt.grid(True)
plt.tight_layout()

plt.savefig(f'./png/{timestamp}_lsa_vs_bucket_mapping_timing.png')
plt.show()

"""
# Plot 3
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, lsa_mean, label='LSA (Average)', color=colors['baseline'], linewidth=2)
plt.fill_between(matrix_sizes, lsa_min, lsa_max, color=colors['baseline'], alpha=0.2, label='LSA Range')

plt.plot(matrix_sizes, min_mean, label='Min Assignment (Average)', color=colors['algorithm3'], linewidth=2)
plt.fill_between(matrix_sizes, min_min, min_max, color=colors['algorithm3'], alpha=0.2,
                 label='Min Assignment Range')

plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Time (seconds)')
plt.title('LSA vs Min Assignment - Runtime Comparison')
plt.legend()
plt.xticks([i for i in range(10, 151, 5)])
plt.ylim(0, None)
plt.xlim(10, 150)
plt.grid(True)
plt.tight_layout()

plt.savefig(f'./png/{timestamp}_lsa_vs_min_assignment_timing.png')
plt.show()

# Plot 4
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, lsa_mean, label='LSA (Average)', color=colors['baseline'], linewidth=2)
plt.fill_between(matrix_sizes, lsa_min, lsa_max, color=colors['baseline'], alpha=0.2, label='LSA Range')

plt.plot(matrix_sizes, divider_mean, label='Matrix Divider (Average)', color=colors['algorithm1'], linewidth=2)
plt.fill_between(matrix_sizes, divider_min, divider_max, color=colors['algorithm1'], alpha=0.2,
                 label='Matrix Divider Range')

plt.plot(matrix_sizes, bucket_mean, label='Bucket Mapping (Average)', color=colors['algorithm2'], linewidth=2)
plt.fill_between(matrix_sizes, bucket_min, bucket_max, color=colors['algorithm2'], alpha=0.2,
                 label='Bucket Mapping Range')

plt.plot(matrix_sizes, min_mean, label='Min Assignment (Average)', color=colors['algorithm3'], linewidth=2)
plt.fill_between(matrix_sizes, min_min, min_max, color=colors['algorithm3'], alpha=0.2,
                 label='Min Assignment Range')

plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Time (seconds)')
plt.title('LSA vs Matrix Divider vs Min Assignment - Runtime Comparison')
plt.legend()
plt.xticks([i for i in range(10, 151, 5)])
plt.ylim(0, None)
plt.xlim(10, 150)
plt.grid(True)
plt.tight_layout()

plt.savefig(f'./png/{timestamp}_lsa_vs_algo_timings.png')
plt.show()
"""