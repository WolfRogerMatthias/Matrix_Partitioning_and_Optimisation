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
import pandas as pd

from sklearn.metrics import r2_score

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
number_of_matrices = 100
colors = {
    'lsa': '#333333',  # Black
    'divider': '#FF8C00',  # Orange
    'bucket': '#800080',  # Purple
    'direct': '#0000FF'  # Blue
}

"""
The evaluation for the Divider Algorithm.

"""
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
matrix_sizes_divider = [i for i in range(10, 81, 5)]

lsa_mapping_divider = []
divider_mapping = []
cost_matrices_divider = []

for matrix_size in matrix_sizes_divider:
    cost_matrices_divider.append(
        MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5', number_of_matrices))
    lsa_mapping_divider.append([OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_divider[-1][i]) for i in
                                range(number_of_matrices)])
    divider_mapping.append(MatrixDivider.divider(cost_matrices_divider[-1], number_of_matrices, 4))

# === Store results for R², relative error, std, F1, and scatter data ===
r2_results_divider = []
scatter_data_divider = []

all_lsa_sums = []
all_divider_sums = []

for size_idx, size in enumerate(matrix_sizes_divider):
    lsa_sums_size = []
    divider_sums_size = []
    rel_errors_per_matrix = []
    f1_per_matrix = []

    for matrix_idx in range(number_of_matrices):
        matrix = cost_matrices_divider[size_idx][matrix_idx]

        # LSA minimal sum
        row_lsa, col_lsa = lsa_mapping_divider[size_idx][matrix_idx]
        lsa_sum = matrix[row_lsa, col_lsa].sum()
        lsa_sums_size.append(lsa_sum)
        all_lsa_sums.append(lsa_sum)

        # Divider minimal sum
        row_div, col_div = divider_mapping[size_idx][matrix_idx]
        divider_sum = matrix[row_div, col_div].sum()
        divider_sums_size.append(divider_sum)
        all_divider_sums.append(divider_sum)

        # Relative error
        rel_error = (divider_sum - lsa_sum) / lsa_sum
        rel_errors_per_matrix.append(rel_error)

        # F1 score
        lsa_positions = set(zip(row_lsa, col_lsa))
        divider_positions = set(zip(row_div, col_div))
        tp = len(lsa_positions & divider_positions)
        fp = len(divider_positions - lsa_positions)
        fn = len(lsa_positions - lsa_positions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_matrix.append(f1)

    # Compute R²
    r2 = r2_score(lsa_sums_size, divider_sums_size)

    # Compute mean relative error and std
    mean_rel_error = np.mean(rel_errors_per_matrix)
    std_rel_error = np.std(rel_errors_per_matrix)

    # Compute mean F1 across matrices
    mean_f1 = np.mean(f1_per_matrix)

    r2_results_divider.append({
        "matrix_size": size,
        "r2": r2,
        "mean_relative_error": mean_rel_error,
        "std_relative_error": std_rel_error,
        "mean_f1": mean_f1
    })

    # Store scatter data
    scatter_data_divider.append({
        "size": size,
        "lsa_sums": lsa_sums_size,
        "divider_sums": divider_sums_size
    })

# === Save full statistics to CSV ===
df_r2 = pd.DataFrame(r2_results_divider)
df_r2.to_csv(f"./csv/{timestamp}_full_stats_divider.csv", index=False)
print("Saved full statistics to CSV for Divider")

# === Scatter plots for selected sizes ===
selected_sizes_divider = [10, 30, 50, 70]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, size in enumerate(selected_sizes_divider):
    ax = axes[idx]
    data = next((d for d in scatter_data_divider if d["size"] == size), None)
    if not data:
        ax.axis('off')
        continue

    lsa_sums, divider_sums = data["lsa_sums"], data["divider_sums"]
    r2_val = next(r['r2'] for r in r2_results_divider if r['matrix_size'] == size)
    rel_err = next(r['mean_relative_error'] for r in r2_results_divider if r['matrix_size'] == size)
    std_err = next(r['std_relative_error'] for r in r2_results_divider if r['matrix_size'] == size)
    mean_f1 = next(r['mean_f1'] for r in r2_results_divider if r['matrix_size'] == size)

    ax.scatter(lsa_sums, divider_sums, alpha=0.6, color='orange')
    min_val, max_val = min(min(lsa_sums), min(divider_sums)), max(max(lsa_sums), max(divider_sums))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    ax.set_title(f"Size {size}\nR²={r2_val:.3f}, RelErr={rel_err:.2%} ± {std_err:.2%}, F1={mean_f1:.3f}")
    ax.set_xlabel("LSA minimal sum")
    ax.set_ylabel("Divider minimal sum")

fig.tight_layout()
plt.savefig(f"./png/{timestamp}_selected_sizes_2x2_scatter_divider.png", dpi=300)
plt.show()

# === Line plots for R², Relative Error, and F1 ===
sizes = [r['matrix_size'] for r in r2_results_divider]
plt.figure(figsize=(8, 5))
plt.plot(sizes, [r['r2'] for r in r2_results_divider], '-o', color='blue', label='R²')
plt.xlabel("Matrix size")
plt.ylabel("R² value")
plt.title("R² across matrix sizes (Divider)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig(f"./png/{timestamp}_r2_line_plot_divider.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(sizes, [r['mean_relative_error'] for r in r2_results_divider],
             yerr=[r['std_relative_error'] for r in r2_results_divider],
             fmt='-s', color='orange', label='Mean ± Std Rel. Error')
plt.xlabel("Matrix size")
plt.ylabel("Relative Error")
plt.title("Mean Relative Error ± Std across matrix sizes (Divider)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.legend()
plt.savefig(f"./png/{timestamp}_relerror_std_line_plot_divider.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(sizes, [r['mean_f1'] for r in r2_results_divider], '-^', color='green', label='Mean F1 Score')
plt.xlabel("Matrix size")
plt.ylabel("Mean F1 Score")
plt.title("Mean F1 Score across matrix sizes (Divider)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 1.05)
plt.legend()
plt.savefig(f"./png/{timestamp}_f1_line_plot_divider.png", dpi=300)
plt.show()

# === Correlation heatmap between all LSA sums and Divider sums ===
all_sums_df = pd.DataFrame({'LSA': all_lsa_sums, 'Divider': all_divider_sums})
corr_matrix = all_sums_df.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap: LSA vs Divider Sums")
plt.savefig(f"./png/{timestamp}_correlation_heatmap_divider.png", dpi=300)
plt.show()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
matrix_sizes_bucket = [i for i in range(100, 401, 20)]

lsa_mapping_bucket = []
bucket_mapping = []
cost_matrices_bucket = []

for matrix_size in matrix_sizes_bucket:
    cost_matrices_bucket.append(
        MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5', number_of_matrices))
    lsa_mapping_bucket.append([OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_bucket[-1][i]) for i in
                               range(number_of_matrices)])
    bucket_mapping.append(ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_bucket[-1], 2))

# === Compute metrics ===
r2_results_bucket = []
scatter_data_bucket = {}

for size_idx, size in enumerate(matrix_sizes_bucket):
    lsa_sums = []
    bucket_sums = []
    rel_errors_per_matrix = []
    f1_per_matrix = []

    for matrix_idx in range(number_of_matrices):
        matrix = cost_matrices_bucket[size_idx][matrix_idx]

        # LSA minimal sum
        row_lsa, col_lsa = lsa_mapping_bucket[size_idx][matrix_idx]
        lsa_sum = matrix[row_lsa, col_lsa].sum()
        lsa_sums.append(lsa_sum)

        # Bucket minimal sum
        bucket_assignments = bucket_mapping[size_idx][matrix_idx]
        row_bucket = np.array([r for r, c in bucket_assignments])
        col_bucket = np.array([c for r, c in bucket_assignments])
        bucket_sum = matrix[row_bucket, col_bucket].sum()
        bucket_sums.append(bucket_sum)

        # Relative error
        rel_error = (bucket_sum - lsa_sum) / lsa_sum
        rel_errors_per_matrix.append(rel_error)

        # F1 score based on mapping positions
        lsa_positions = set(zip(row_lsa, col_lsa))
        bucket_positions = set(zip(row_bucket, col_bucket))
        tp = len(lsa_positions & bucket_positions)
        fp = len(bucket_positions - lsa_positions)
        fn = len(lsa_positions - bucket_positions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_matrix.append(f1)

    # Compute R²
    r2 = r2_score(lsa_sums, bucket_sums)

    # Compute mean relative error and std
    mean_rel_error = np.mean(rel_errors_per_matrix)
    std_rel_error = np.std(rel_errors_per_matrix)

    # Compute mean F1 across matrices
    mean_f1 = np.mean(f1_per_matrix)

    r2_results_bucket.append({
        "matrix_size": size,
        "r2": r2,
        "mean_relative_error": mean_rel_error,
        "std_relative_error": std_rel_error,
        "mean_f1": mean_f1
    })

    # Store scatter data for plotting
    scatter_data_bucket[size] = (lsa_sums, bucket_sums)

# Save results to CSV
df_r2 = pd.DataFrame(r2_results_bucket)
df_r2.to_csv(f"./csv/{timestamp}_full_stats_bucket.csv", index=False)
print("Saved full statistics to CSV for Bucket Solver")

# === Scatter plots for selected sizes ===
selected_sizes_bucket = [180, 240, 300, 360]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, size in enumerate(selected_sizes_bucket):
    ax = axes[idx]
    if size not in scatter_data_bucket:
        ax.axis('off')
        continue

    lsa_sums, bucket_sums = scatter_data_bucket[size]
    r2 = next(r['r2'] for r in r2_results_bucket if r['matrix_size'] == size)
    rel_err = next(r['mean_relative_error'] for r in r2_results_bucket if r['matrix_size'] == size)
    std_err = next(r['std_relative_error'] for r in r2_results_bucket if r['matrix_size'] == size)
    mean_f1 = next(r['mean_f1'] for r in r2_results_bucket if r['matrix_size'] == size)

    ax.scatter(lsa_sums, bucket_sums, alpha=0.6, color=colors['bucket'])
    min_val = min(min(lsa_sums), min(bucket_sums))
    max_val = max(max(lsa_sums), max(bucket_sums))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    ax.set_title(f"Size {size}\nR²={r2:.3f}, RelErr={rel_err:.2%} ± {std_err:.2%}, F1={mean_f1:.3f}")
    ax.set_xlabel("LSA minimal sum", color=colors['lsa'])
    ax.set_ylabel("Bucket minimal sum", color=colors['bucket'])

fig.tight_layout()
plt.savefig(f"./png/{timestamp}_selected_sizes_2x2_scatter_bucket.png", dpi=300)
plt.show()

# === Line plot for R² values ===
sizes = [r['matrix_size'] for r in r2_results_bucket]
r2_vals = [r['r2'] for r in r2_results_bucket]

plt.figure(figsize=(8, 5))
plt.plot(sizes, r2_vals, marker='o', color=colors['bucket'], label='Bucket R²')
plt.xlabel("Matrix size", color=colors['lsa'])
plt.ylabel("R² value", color=colors['bucket'])
plt.title("R² values across matrix sizes (Bucket Solver)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig(f"./png/{timestamp}_r2_line_plot_bucket.png", dpi=300)
plt.show()

# === Line plot for Mean Relative Error ± Std ===
mean_errors = [r['mean_relative_error'] for r in r2_results_bucket]
std_errors = [r['std_relative_error'] for r in r2_results_bucket]

plt.figure(figsize=(8, 5))
plt.errorbar(sizes, mean_errors, yerr=std_errors, fmt='-s', color='orange', label='Mean ± Std Rel. Error')
plt.xlabel("Matrix size", color=colors['lsa'])
plt.ylabel("Relative Error", color='orange')
plt.title("Mean Relative Error ± Std across matrix sizes (Bucket Solver)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.legend()
plt.savefig(f"./png/{timestamp}_relerror_std_line_plot_bucket.png", dpi=300)
plt.show()

# === Line plot for Mean F1 Score ===
mean_f1_vals = [r['mean_f1'] for r in r2_results_bucket]

plt.figure(figsize=(8, 5))
plt.plot(sizes, mean_f1_vals, marker='^', color='green', label='Mean F1 Score')
plt.xlabel("Matrix size", color=colors['lsa'])
plt.ylabel("Mean F1 Score", color='green')
plt.title("Mean F1 Score across matrix sizes (Bucket Solver)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 1.05)
plt.legend()
plt.savefig(f"./png/{timestamp}_f1_line_plot_bucket.png", dpi=300)
plt.show()

# === Heatmap for LSA vs Bucket sums correlation ===
heatmap_matrix = np.zeros((len(matrix_sizes_bucket), len(matrix_sizes_bucket)))

for i, size_i in enumerate(matrix_sizes_bucket):
    _, sums_i = scatter_data_bucket[size_i]
    for j, size_j in enumerate(matrix_sizes_bucket):
        _, sums_j = scatter_data_bucket[size_j]
        heatmap_matrix[i, j] = np.corrcoef(sums_i, sums_j)[0, 1]

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_matrix, annot=True, xticklabels=matrix_sizes_bucket, yticklabels=matrix_sizes_bucket,
            cmap="coolwarm", vmin=0, vmax=1)
plt.title("Correlation Heatmap: Bucket vs LSA sums")
plt.xlabel("Matrix size")
plt.ylabel("Matrix size")
plt.savefig(f"./png/{timestamp}_heatmap_bucket.png", dpi=300)
plt.show()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
matrix_sizes_direct = [i for i in range(10, 151, 5)]

lsa_mapping_direct = []
direct_mapping = []
cost_matrices_direct = []

for matrix_size in matrix_sizes_direct:
    cost_matrices_direct.append(
        MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5', number_of_matrices))
    lsa_mapping_direct.append([OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_direct[-1][i]) for i in
                               range(number_of_matrices)])
    direct_mapping.append(
        [AssignmentAlgo.assignment_applied(cost_matrices_direct[-1][i]) for i in range(number_of_matrices)])

# === Compute metrics for Direct Assignment ===
r2_results_direct = []
scatter_data_direct = {}

for size_idx, size in enumerate(matrix_sizes_direct):
    lsa_sums = []
    direct_sums = []
    rel_errors_per_matrix = []
    f1_per_matrix = []

    for matrix_idx in range(number_of_matrices):
        matrix = cost_matrices_direct[size_idx][matrix_idx]

        # LSA minimal sum
        row_lsa, col_lsa = lsa_mapping_direct[size_idx][matrix_idx]
        lsa_sum = matrix[row_lsa, col_lsa].sum()
        lsa_sums.append(lsa_sum)

        # Direct minimal sum
        direct_assignments = direct_mapping[size_idx][matrix_idx]
        row_direct = np.array(list(direct_assignments.keys()))
        col_direct = np.array(list(direct_assignments.values()))
        direct_sum = matrix[row_direct, col_direct].sum()
        direct_sums.append(direct_sum)

        # Relative error
        rel_error = (direct_sum - lsa_sum) / lsa_sum
        rel_errors_per_matrix.append(rel_error)

        # F1 score based on mapping positions
        lsa_positions = set(zip(row_lsa, col_lsa))
        direct_positions = set(zip(row_direct, col_direct))
        tp = len(lsa_positions & direct_positions)
        fp = len(direct_positions - lsa_positions)
        fn = len(lsa_positions - direct_positions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_matrix.append(f1)

    # Compute R²
    r2 = r2_score(lsa_sums, direct_sums)

    # Compute mean relative error and std
    mean_rel_error = np.mean(rel_errors_per_matrix)
    std_rel_error = np.std(rel_errors_per_matrix)

    # Compute mean F1 across matrices
    mean_f1 = np.mean(f1_per_matrix)

    r2_results_direct.append({
        "matrix_size": size,
        "r2": r2,
        "mean_relative_error": mean_rel_error,
        "std_relative_error": std_rel_error,
        "mean_f1": mean_f1
    })

    # Store scatter data for plotting
    scatter_data_direct[size] = (lsa_sums, direct_sums)

# Save results to CSV
df_r2_direct = pd.DataFrame(r2_results_direct)
df_r2_direct.to_csv(f"./csv/{timestamp}_full_stats_direct.csv", index=False)
print("Saved full statistics to CSV for Direct Assignment Solver")

# === Scatter plots for selected sizes ===
selected_sizes_direct = [70, 90, 110, 130]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, size in enumerate(selected_sizes_direct):
    ax = axes[idx]
    if size not in scatter_data_direct:
        ax.axis('off')
        continue

    lsa_sums, direct_sums = scatter_data_direct[size]
    r2 = next(r['r2'] for r in r2_results_direct if r['matrix_size'] == size)
    rel_err = next(r['mean_relative_error'] for r in r2_results_direct if r['matrix_size'] == size)
    std_err = next(r['std_relative_error'] for r in r2_results_direct if r['matrix_size'] == size)
    mean_f1 = next(r['mean_f1'] for r in r2_results_direct if r['matrix_size'] == size)

    ax.scatter(lsa_sums, direct_sums, alpha=0.6, color=colors['direct'])
    min_val = min(min(lsa_sums), min(direct_sums))
    max_val = max(max(lsa_sums), max(direct_sums))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)

    ax.set_title(f"Size {size}\nR²={r2:.3f}, RelErr={rel_err:.2%} ± {std_err:.2%}, F1={mean_f1:.3f}")
    ax.set_xlabel("LSA minimal sum", color=colors['lsa'])
    ax.set_ylabel("Direct minimal sum", color=colors['direct'])

fig.tight_layout()
plt.savefig(f"./png/{timestamp}_selected_sizes_2x2_scatter_direct.png", dpi=300)
plt.show()

# === Line plots ===
sizes = [r['matrix_size'] for r in r2_results_direct]

# R² line plot
r2_vals = [r['r2'] for r in r2_results_direct]
plt.figure(figsize=(8, 5))
plt.plot(sizes, r2_vals, marker='o', color=colors['direct'], label='Direct R²')
plt.xlabel("Matrix size", color=colors['lsa'])
plt.ylabel("R² value", color=colors['direct'])
plt.title("R² values across matrix sizes (Direct Solver)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig(f"./png/{timestamp}_r2_line_plot_direct.png", dpi=300)
plt.show()

# Mean Relative Error ± Std
mean_errors = [r['mean_relative_error'] for r in r2_results_direct]
std_errors = [r['std_relative_error'] for r in r2_results_direct]
plt.figure(figsize=(8, 5))
plt.errorbar(sizes, mean_errors, yerr=std_errors, fmt='-s', color='orange', label='Mean ± Std Rel. Error')
plt.xlabel("Matrix size")
plt.ylabel("Relative Error")
plt.title("Mean Relative Error ± Std across matrix sizes (Direct Solver)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='gray', linestyle='--', lw=1)
plt.legend()
plt.savefig(f"./png/{timestamp}_relerror_std_line_plot_direct.png", dpi=300)
plt.show()

# Mean F1 line plot
mean_f1_vals = [r['mean_f1'] for r in r2_results_direct]
plt.figure(figsize=(8, 5))
plt.plot(sizes, mean_f1_vals, marker='^', color='green', label='Mean F1 Score')
plt.xlabel("Matrix size")
plt.ylabel("Mean F1 Score")
plt.title("Mean F1 Score across matrix sizes (Direct Solver)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 1.05)
plt.legend()
plt.savefig(f"./png/{timestamp}_f1_line_plot_direct.png", dpi=300)
plt.show()

# Heatmap: correlation between different sizes
heatmap_matrix = np.zeros((len(matrix_sizes_direct), len(matrix_sizes_direct)))
for i, size_i in enumerate(matrix_sizes_direct):
    _, sums_i = scatter_data_direct[size_i]
    for j, size_j in enumerate(matrix_sizes_direct):
        _, sums_j = scatter_data_direct[size_j]
        heatmap_matrix[i, j] = np.corrcoef(sums_i, sums_j)[0, 1]

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_matrix, annot=False, xticklabels=matrix_sizes_direct, yticklabels=matrix_sizes_direct,
            cmap="coolwarm", vmin=0, vmax=1)
plt.title("Correlation Heatmap: Direct vs LSA sums")
plt.xlabel("Matrix size")
plt.ylabel("Matrix size")
plt.savefig(f"./png/{timestamp}_heatmap_direct.png", dpi=300)
plt.show()
