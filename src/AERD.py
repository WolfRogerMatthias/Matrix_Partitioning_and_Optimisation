"""
Accuracy evaluation of the Real world Datasets
"""

# Imports
import datetime
import time
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tkinter as tk
from pandastable import Table

from itertools import combinations, chain
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.AssignmentAlgo import AssignmentAlgo
from src.MatrixDivider import MatrixDivider
from src.MockDataGenerator import MockDataGenerator
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver

# init

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
colors = {
    'Matrix Divider': '#FF8C00',  # Orange
    'Bucket': '#800080',          # Purple
    'Direct': '#0000FF'           # Blue
}
ALGO_NAME_MAP = {
    'divider': 'Matrix Divider',
    'bucket': 'Bucket',
    'direct': 'Direct'
}

len_mutag = 188
len_ohsu = 79

number_of_matrices_mutag = len(list(combinations(range(len_mutag), r=2)))
number_of_matrices_ohsu = len(list(combinations(range(len_ohsu), r=2)))
number_of_matrices_proteins = 10000

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
AssignmentAlgo = AssignmentAlgo()
MatrixDivider = MatrixDivider()
ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()

start = time.time()
cost_matrices_mutag = MockDataGenerator.load_h5_file('./data/cost_matrices.h5', number_of_matrices_mutag)
cost_matrices_ohsu = MockDataGenerator.load_h5_file('./data/cost_matrices_OHSU.h5', number_of_matrices_ohsu)
cost_matrices_proteins = MockDataGenerator.load_h5_file('./data/cost_matrices_PROTEINS_subset.h5',
                                                        number_of_matrices_proteins)

print(f'Loading of Datasets done time:{time.time() - start:6.2f}')
start = time.time()

lsa_mapping_mutag = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_mutag[j]) for j in
                     range(number_of_matrices_mutag)]
lsa_mapping_ohsu = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_ohsu[j]) for j in
                    range(number_of_matrices_ohsu)]
lsa_mapping_proteins = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_proteins[j]) for j in
                        range(number_of_matrices_proteins)]

print(f'Computing of Lsa done time:{time.time() - start:6.2f}')
start = time.time()

divider_mapping_mutag = MatrixDivider.divider(cost_matrices_mutag, number_of_matrices_mutag, 4)
divider_mapping_ohsu = MatrixDivider.divider(cost_matrices_ohsu, number_of_matrices_ohsu, 4)
divider_mapping_proteins = MatrixDivider.divider(cost_matrices_proteins, number_of_matrices_proteins, 4)

print(f'Computing of divider done time{time.time() - start:6.2f}')
start = time.time()

bucket_mapping_mutag = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_mutag, 2)
bucket_mapping_ohsu = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_ohsu, 2)
bucket_mapping_proteins = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_proteins, 2)

print(f'Computing of bucket done time{time.time() - start:6.2f}')
start = time.time()

direct_mapping_mutag = [AssignmentAlgo.assignment_applied(cost_matrices_mutag[j]) for j in
                        range(number_of_matrices_mutag)]
direct_mapping_ohsu = [AssignmentAlgo.assignment_applied(cost_matrices_ohsu[j]) for j in range(number_of_matrices_ohsu)]
direct_mapping_proteins = [AssignmentAlgo.assignment_applied(cost_matrices_proteins[j]) for j in
                           range(number_of_matrices_proteins)]

print(f'Computing of direct mapping done time{time.time() - start:6.2f}')

data_cost_gap = {
    'divider_mutag': [],
    'divider_proteins': [],
    'divider_ohsu': [],
    'bucket_mutag': [],
    'bucket_proteins': [],
    'bucket_ohsu': [],
    'direct_mutag': [],
    'direct_proteins': [],
    'direct_ohsu': [],
}


def cost_gap_evaluation(cost_matrices, optimal_mapping, approximate_mapping, algo, dataset):
    for i in range(len(cost_matrices)):
        matrix = cost_matrices[i]
        opt_mapping_x = optimal_mapping[i][0]
        opt_mapping_y = optimal_mapping[i][1]
        opt_sum = sum(matrix[opt_mapping_x, opt_mapping_y])
        app_sum = 0
        if algo == 'divider':
            app_mapping_x = approximate_mapping[i][0]
            app_mapping_y = approximate_mapping[i][1]
            app_sum = matrix[app_mapping_x, app_mapping_y].sum()
        if algo == 'bucket':
            app_mapping = approximate_mapping[i]
            app_sum = sum([matrix[x, y] for x, y in app_mapping])
        if algo == 'direct':
            app_mapping = approximate_mapping[i]
            app_sum = sum([matrix[row, col] for row, col in app_mapping.items()])

        cost_gap = (app_sum - opt_sum) / (opt_sum + 1e-6)
        data_cost_gap[f'{algo}_{dataset}'].append(cost_gap)


cost_gap_evaluation(cost_matrices_mutag, lsa_mapping_mutag, divider_mapping_mutag, 'divider', 'mutag')
cost_gap_evaluation(cost_matrices_mutag, lsa_mapping_mutag, bucket_mapping_mutag, 'bucket', 'mutag')
cost_gap_evaluation(cost_matrices_mutag, lsa_mapping_mutag, direct_mapping_mutag, 'direct', 'mutag')

cost_gap_evaluation(cost_matrices_proteins, lsa_mapping_proteins, divider_mapping_proteins, 'divider', 'proteins')
cost_gap_evaluation(cost_matrices_proteins, lsa_mapping_proteins, bucket_mapping_proteins, 'bucket', 'proteins')
cost_gap_evaluation(cost_matrices_proteins, lsa_mapping_proteins, direct_mapping_proteins, 'direct', 'proteins')

cost_gap_evaluation(cost_matrices_ohsu, lsa_mapping_ohsu, divider_mapping_ohsu, 'divider', 'ohsu')
cost_gap_evaluation(cost_matrices_ohsu, lsa_mapping_ohsu, bucket_mapping_ohsu, 'bucket', 'ohsu')
cost_gap_evaluation(cost_matrices_ohsu, lsa_mapping_ohsu, direct_mapping_ohsu, 'direct', 'ohsu')

summary_cost_gap = {}

for key, values in data_cost_gap.items():
    arr = np.array(values)
    algo, dataset = key.split('_')

    if algo not in summary_cost_gap:
        summary_cost_gap[algo] = {}

    summary_cost_gap[algo][f"{dataset}_mean"] = np.mean(arr)
    summary_cost_gap[algo][f"{dataset}_std"] = np.std(arr)
    summary_cost_gap[algo][f"{dataset}_median"] = np.median(arr)

# Convert to DataFrame (algorithms as rows, metrics as columns)
df_summary_cost_gap = pd.DataFrame(summary_cost_gap).T.reset_index().rename(columns={"index": "Algorithm"})

# Save to CSV
df_summary_cost_gap.to_csv(f'./csv/{timestamp}_cost_gap_summary.csv', index=False)

print(df_summary_cost_gap)

data_acc = {
    'divider_mutag': [],
    'divider_proteins': [],
    'divider_ohsu': [],
    'bucket_mutag': [],
    'bucket_proteins': [],
    'bucket_ohsu': [],
    'direct_mutag': [],
    'direct_proteins': [],
    'direct_ohsu': [],
}


def acc_evaluation(cost_matrix, optimal_mapping, approximate_mapping, algo, dataset):
    for i in range(len(cost_matrix)):
        matrix = cost_matrix[i]
        opt_mapping_x = optimal_mapping[i][0]
        opt_mapping_y = optimal_mapping[i][1]
        count = 0
        n = len(opt_mapping_x)
        if algo == 'divider':
            app_mapping_x = approximate_mapping[i][0]
            app_mapping_y = approximate_mapping[i][1]
            for k in range(n):
                count += 1 if (opt_mapping_x[k] == app_mapping_x[k] and opt_mapping_y[k] == app_mapping_y[k]) else 0
        if algo == 'bucket':
            app_mapping = approximate_mapping[i]
            for k in range(n):
                count += 1 if (opt_mapping_x[k] == app_mapping[k][0] and opt_mapping_y[k] == app_mapping[k][1]) else 0
        if algo == 'direct':
            app_mapping = approximate_mapping[i]
            for k in range(n):
                count += 1 if (opt_mapping_y[k] == app_mapping[k]) else 0
        acc = count / n
        data_acc[f'{algo}_{dataset}'].append(acc)


acc_evaluation(cost_matrices_mutag, lsa_mapping_mutag, divider_mapping_mutag, 'divider', 'mutag')
acc_evaluation(cost_matrices_mutag, lsa_mapping_mutag, bucket_mapping_mutag, 'bucket', 'mutag')
acc_evaluation(cost_matrices_mutag, lsa_mapping_mutag, direct_mapping_mutag, 'direct', 'mutag')

acc_evaluation(cost_matrices_proteins, lsa_mapping_proteins, divider_mapping_proteins, 'divider', 'proteins')
acc_evaluation(cost_matrices_proteins, lsa_mapping_proteins, bucket_mapping_proteins, 'bucket', 'proteins')
acc_evaluation(cost_matrices_proteins, lsa_mapping_proteins, direct_mapping_proteins, 'direct', 'proteins')

acc_evaluation(cost_matrices_ohsu, lsa_mapping_ohsu, divider_mapping_ohsu, 'divider', 'ohsu')
acc_evaluation(cost_matrices_ohsu, lsa_mapping_ohsu, bucket_mapping_ohsu, 'bucket', 'ohsu')
acc_evaluation(cost_matrices_ohsu, lsa_mapping_ohsu, direct_mapping_ohsu, 'direct', 'ohsu')

summary_acc = {}

for key, values in data_acc.items():
    arr = np.array(values)
    algo, dataset = key.split('_')

    if algo not in summary_acc:
        summary_acc[algo] = {}

    summary_acc[algo][f"{dataset}_mean"] = np.mean(arr)
    summary_acc[algo][f"{dataset}_std"] = np.std(arr)
    summary_acc[algo][f"{dataset}_median"] = np.median(arr)

# Convert to DataFrame (algorithms as rows, metrics as columns)
df_summary_acc = pd.DataFrame(summary_acc).T.reset_index().rename(columns={"index": "Algorithm"})

# Save to CSV
df_summary_acc.to_csv(f'./csv/{timestamp}_acc_summary.csv', index=False)

print(df_summary_acc)

# Common settings
figsize = (8, 5)
palette = colors
sns.set(style="whitegrid", context="talk", font_scale=1.1)


def plot_bar(df_plot, y_col, title, ylabel, log_scale=False, save_name=None):
    """
    Plots a barplot with algorithm names automatically renamed.
    """
    # Rename algorithms using the global mapping
    df_plot['Algorithm'] = df_plot['Algorithm'].map(ALGO_NAME_MAP)

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_plot, x='Dataset', y=y_col, hue='Algorithm', palette=colors)

    if log_scale:
        plt.yscale('log')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('')  # remove x-axis label

    # Move legend outside plot
    plt.legend(title="Algorithm", loc='center left', bbox_to_anchor=(1, 0.5))

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')

    # plt.show()


# -------------------------------
# Median Cost Gap
# -------------------------------
median_cols = [col for col in df_summary_cost_gap.columns if 'median' in col]
df_median_cost_gap = df_summary_cost_gap[['Algorithm'] + median_cols]
df_median_cost_gap_plot = df_median_cost_gap.melt(id_vars='Algorithm', var_name='Dataset', value_name='Median')
df_median_cost_gap_plot['Dataset'] = df_median_cost_gap_plot['Dataset'].str.replace('_median', '').str.upper()

plot_bar(df_median_cost_gap_plot,
         y_col='Median',
         title="Median Cost Gap by Algorithm and Dataset",
         ylabel="Median Cost Gap",
         save_name=f'./png/cost_gap_median_barplot_log.png')

# -------------------------------
# Repeat for the others
# -------------------------------

# Median Accuracy
median_cols = [col for col in df_summary_acc.columns if 'median' in col]
df_median_acc = df_summary_acc[['Algorithm'] + median_cols]
df_median_acc_plot = df_median_acc.melt(id_vars='Algorithm', var_name='Dataset', value_name='Median')
df_median_acc_plot['Dataset'] = df_median_acc_plot['Dataset'].str.replace('_median', '').str.upper()

plot_bar(df_median_acc_plot,
         y_col='Median',
         title="Median Accuracy by Algorithm and Dataset",
         ylabel="Median Accuracy",
         log_scale=False,
         save_name=f'./png/acc_median_barplot.png')

# Mean Cost Gap
mean_cols = [col for col in df_summary_cost_gap.columns if 'mean' in col]
df_mean_cost_gap = df_summary_cost_gap[['Algorithm'] + mean_cols]
df_mean_cost_gap_plot = df_mean_cost_gap.melt(id_vars='Algorithm', var_name='Dataset', value_name='Mean')
df_mean_cost_gap_plot['Dataset'] = df_mean_cost_gap_plot['Dataset'].str.replace('_mean', '').str.upper()

plot_bar(df_mean_cost_gap_plot,
         y_col='Mean',
         title="Mean Cost Gap by Algorithm and Dataset",
         ylabel="Mean Cost Gap (log scale)",
         log_scale=True,
         save_name=f'./png/cost_gap_mean_barplot_log.png')

# Mean Accuracy
mean_cols = [col for col in df_summary_acc.columns if 'mean' in col]
df_mean_acc = df_summary_acc[['Algorithm'] + mean_cols]
df_mean_acc_plot = df_mean_acc.melt(id_vars='Algorithm', var_name='Dataset', value_name='Mean')
df_mean_acc_plot['Dataset'] = df_mean_acc_plot['Dataset'].str.replace('_mean', '').str.upper()

plot_bar(df_mean_acc_plot,
         y_col='Mean',
         title="Mean Accuracy by Algorithm and Dataset",
         ylabel="Mean Accuracy",
         log_scale=False,
         save_name=f'./png/acc_mean_barplot.png')
