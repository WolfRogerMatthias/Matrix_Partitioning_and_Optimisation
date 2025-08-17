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
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import r2_score

# init

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
colors = {
    'lsa': '#333333',  # Black
    'divider': '#FF8C00',  # Orange
    'bucket': '#800080',  # Purple
    'direct': '#0000FF'  # Blue
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

cost_matrices_mutag = MockDataGenerator.load_h5_file('./data/cost_matrices.h5', number_of_matrices_mutag)
cost_matrices_ohsu = MockDataGenerator.load_h5_file('./data/cost_matrices_OHSU.h5', number_of_matrices_ohsu)
cost_matrices_proteins = MockDataGenerator.load_h5_file('./data/cost_matrices_PROTEINS_subset.h5',
                                                        number_of_matrices_proteins)

lsa_mapping_mutag = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_mutag[j]) for j in
                     range(number_of_matrices_mutag)]
lsa_mapping_ohsu = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_ohsu[j]) for j in
                    range(number_of_matrices_ohsu)]
lsa_mapping_proteins = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_proteins[j]) for j in
                        range(number_of_matrices_proteins)]

divider_mapping_mutag = MatrixDivider.divider(cost_matrices_mutag, number_of_matrices_mutag, 4)
divider_mapping_ohsu = MatrixDivider.divider(cost_matrices_ohsu, number_of_matrices_ohsu, 4)
divider_mapping_proteins = MatrixDivider.divider(cost_matrices_proteins, number_of_matrices_proteins, 4)

bucket_mapping_mutag = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_mutag, 2)
bucket_mapping_ohsu = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_ohsu, 2)
bucket_mapping_proteins = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_proteins, 2)

direct_mapping_mutag = [AssignmentAlgo.assignment_applied(cost_matrices_mutag[j]) for j in
                        range(number_of_matrices_mutag)]
direct_mapping_ohsu = [AssignmentAlgo.assignment_applied(cost_matrices_ohsu[j]) for j in range(number_of_matrices_ohsu)]
direct_mapping_proteins = [AssignmentAlgo.assignment_applied(cost_matrices_proteins[j]) for j in
                           range(number_of_matrices_proteins)]


def compute_metrics(baseline_pairs, assigned_pairs):
    tp = len(baseline_pairs.intersection(assigned_pairs))
    fp = len(assigned_pairs - baseline_pairs)
    fn = len(baseline_pairs - assigned_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision * 100, recall * 100, f1 * 100  # percentages


def compute_agreement_score(baseline_pairs, assigned_pairs):  # NEW
    return (len(baseline_pairs & assigned_pairs) /
            len(baseline_pairs | assigned_pairs)) * 100 if baseline_pairs or assigned_pairs else 0.0


def compute_optimality_gap(cost_sum, optimal_sum):  # NEW
    return ((cost_sum - optimal_sum) / optimal_sum) * 100 if optimal_sum != 0 else 0.0


def compute_rank_correlation(matrix, baseline_pairs, test_pairs):
    
    # Compares the cost rankings of the assignments in the baseline vs test.
    # Returns Spearman and Kendall correlation in percentage.
    
    baseline_costs = [matrix[r, c] for r, c in baseline_pairs]
    test_costs = [matrix[r, c] for r, c in test_pairs]

    # Pad if needed in case of missing pairs
    min_len = min(len(baseline_costs), len(test_costs))
    baseline_costs = baseline_costs[:min_len]
    test_costs = test_costs[:min_len]

    spearman_corr, _ = spearmanr(baseline_costs, test_costs)
    kendall_corr, _ = kendalltau(baseline_costs, test_costs)

    return spearman_corr * 100 if spearman_corr is not None else 0.0, \
        kendall_corr * 100 if kendall_corr is not None else 0.0


def compute_r2_for_min_sums(min_sums_dict):
    
    # Computes R² between each algo's min sums and LSA's min sums.
    # Returns a dict: algo -> R² value.
    
    results = {}
    lsa_values = np.array(min_sums_dict['lsa'])
    for algo in ['divider', 'bucket', 'direct']:
        algo_values = np.array(min_sums_dict[algo])
        results[algo] = r2_score(lsa_values, algo_values) * 100
    return results


# For MUTAG
mutag_min_sums = {'lsa': [], 'divider': [], 'bucket': [], 'direct': []}
mutag_precision = {'divider': [], 'bucket': [], 'direct': []}
mutag_recall = {'divider': [], 'bucket': [], 'direct': []}
mutag_f1 = {'divider': [], 'bucket': [], 'direct': []}
mutag_gap = {'divider': [], 'bucket': [], 'direct': []}
mutag_agreement = {'divider': [], 'bucket': [], 'direct': []}
mutag_spearman = {'divider': [], 'bucket': [], 'direct': []}
mutag_kendall = {'divider': [], 'bucket': [], 'direct': []}

# --- For MUTAG ---
for idx, key in enumerate(cost_matrices_mutag.keys()):
    matrix = cost_matrices_mutag[key]
    lsa_rows, lsa_cols = lsa_mapping_mutag[idx]
    baseline_pairs = set(zip(lsa_rows, lsa_cols))
    lsa_sum = matrix[lsa_rows, lsa_cols].sum()
    mutag_min_sums['lsa'].append(lsa_sum)

    # Divider
    div_rows, div_cols = divider_mapping_mutag[idx]
    div_pairs = set(zip(div_rows, div_cols))
    div_sum = matrix[div_rows, div_cols].sum()
    mutag_min_sums['divider'].append(div_sum)
    p, r, f = compute_metrics(baseline_pairs, div_pairs)
    mutag_precision['divider'].append(p)
    mutag_recall['divider'].append(r)
    mutag_f1['divider'].append(f)
    mutag_gap['divider'].append(compute_optimality_gap(div_sum, lsa_sum))  # NEW
    mutag_agreement['divider'].append(compute_agreement_score(baseline_pairs, div_pairs))  # NEW
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, div_pairs)
    mutag_spearman['divider'].append(sp)
    mutag_kendall['divider'].append(kt)

    # Bucket
    bucket_pairs = set(bucket_mapping_mutag[idx])
    bucket_sum = sum(matrix[r, c] for r, c in bucket_pairs)
    mutag_min_sums['bucket'].append(bucket_sum)
    p, r, f = compute_metrics(baseline_pairs, bucket_pairs)
    mutag_precision['bucket'].append(p)
    mutag_recall['bucket'].append(r)
    mutag_f1['bucket'].append(f)
    mutag_gap['bucket'].append(compute_optimality_gap(bucket_sum, lsa_sum))  # NEW
    mutag_agreement['bucket'].append(compute_agreement_score(baseline_pairs, bucket_pairs))  # NEW
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, bucket_pairs)
    mutag_spearman['bucket'].append(sp)
    mutag_kendall['bucket'].append(kt)

    # Direct
    direct_pairs = set(direct_mapping_mutag[idx].items())
    direct_sum = sum(matrix[r, c] for r, c in direct_pairs)
    mutag_min_sums['direct'].append(direct_sum)
    p, r, f = compute_metrics(baseline_pairs, direct_pairs)
    mutag_precision['direct'].append(p)
    mutag_recall['direct'].append(r)
    mutag_f1['direct'].append(f)
    mutag_gap['direct'].append(compute_optimality_gap(direct_sum, lsa_sum))  # NEW
    mutag_agreement['direct'].append(compute_agreement_score(baseline_pairs, direct_pairs))  # NEW
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, direct_pairs)
    mutag_spearman['direct'].append(sp)
    mutag_kendall['direct'].append(kt)

# For OHSU
ohsu_min_sums = {'lsa': [], 'divider': [], 'bucket': [], 'direct': []}
ohsu_precision = {'divider': [], 'bucket': [], 'direct': []}
ohsu_recall = {'divider': [], 'bucket': [], 'direct': []}
ohsu_f1 = {'divider': [], 'bucket': [], 'direct': []}
ohsu_gap = {'divider': [], 'bucket': [], 'direct': []}
ohsu_agreement = {'divider': [], 'bucket': [], 'direct': []}
ohsu_spearman = {'divider': [], 'bucket': [], 'direct': []}
ohsu_kendall = {'divider': [], 'bucket': [], 'direct': []}

for idx, key in enumerate(cost_matrices_ohsu.keys()):
    matrix = cost_matrices_ohsu[key]
    lsa_rows, lsa_cols = lsa_mapping_ohsu[idx]
    baseline_pairs = set(zip(lsa_rows, lsa_cols))
    ohsu_min_sums['lsa'].append(matrix[lsa_rows, lsa_cols].sum())

    div_rows, div_cols = divider_mapping_ohsu[idx]
    div_pairs = set(zip(div_rows, div_cols))
    ohsu_min_sums['divider'].append(matrix[div_rows, div_cols].sum())
    p, r, f = compute_metrics(baseline_pairs, div_pairs)
    ohsu_precision['divider'].append(p)
    ohsu_recall['divider'].append(r)
    ohsu_f1['divider'].append(f)
    ohsu_gap['divider'].append(compute_agreement_score(baseline_pairs, div_pairs))
    ohsu_agreement['divider'].append(compute_agreement_score(baseline_pairs, div_pairs))
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, div_pairs)
    ohsu_spearman['divider'].append(sp)
    ohsu_kendall['divider'].append(kt)

    bucket_pairs = set(bucket_mapping_ohsu[idx])
    bucket_sum = sum(matrix[r, c] for r, c in bucket_pairs)
    ohsu_min_sums['bucket'].append(bucket_sum)
    p, r, f = compute_metrics(baseline_pairs, bucket_pairs)
    ohsu_precision['bucket'].append(p)
    ohsu_recall['bucket'].append(r)
    ohsu_f1['bucket'].append(f)
    ohsu_gap['bucket'].append(compute_agreement_score(baseline_pairs, bucket_pairs))
    ohsu_agreement['bucket'].append(compute_agreement_score(baseline_pairs, bucket_pairs))
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, bucket_pairs)
    ohsu_spearman['bucket'].append(sp)
    ohsu_kendall['bucket'].append(kt)

    direct_assignment = direct_mapping_ohsu[idx]
    direct_pairs = set(direct_assignment.items())
    direct_sum = sum(matrix[r, c] for r, c in direct_pairs)
    ohsu_min_sums['direct'].append(direct_sum)
    p, r, f = compute_metrics(baseline_pairs, direct_pairs)
    ohsu_precision['direct'].append(p)
    ohsu_recall['direct'].append(r)
    ohsu_f1['direct'].append(f)
    ohsu_gap['direct'].append(compute_agreement_score(baseline_pairs, direct_pairs))
    ohsu_agreement['direct'].append(compute_agreement_score(baseline_pairs, direct_pairs))
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, direct_pairs)
    ohsu_spearman['direct'].append(sp)
    ohsu_kendall['direct'].append(kt)

# Fort PROTEINS
proteins_min_sums = {'lsa': [], 'divider': [], 'bucket': [], 'direct': []}
proteins_precision = {'divider': [], 'bucket': [], 'direct': []}
proteins_recall = {'divider': [], 'bucket': [], 'direct': []}
proteins_f1 = {'divider': [], 'bucket': [], 'direct': []}
proteins_gap = {'divider': [], 'bucket': [], 'direct': []}
proteins_agreement = {'divider': [], 'bucket': [], 'direct': []}
proteins_spearman = {'divider': [], 'bucket': [], 'direct': []}
proteins_kendall = {'divider': [], 'bucket': [], 'direct': []}

for idx, key in enumerate(cost_matrices_proteins.keys()):
    matrix = cost_matrices_proteins[key]
    lsa_rows, lsa_cols = lsa_mapping_proteins[idx]
    baseline_pairs = set(zip(lsa_rows, lsa_cols))
    proteins_min_sums['lsa'].append(matrix[lsa_rows, lsa_cols].sum())

    div_rows, div_cols = divider_mapping_proteins[idx]
    div_pairs = set(zip(div_rows, div_cols))
    proteins_min_sums['divider'].append(matrix[div_rows, div_cols].sum())
    p, r, f = compute_metrics(baseline_pairs, div_pairs)
    proteins_precision['divider'].append(p)
    proteins_recall['divider'].append(r)
    proteins_f1['divider'].append(f)
    proteins_gap['divider'].append(compute_agreement_score(baseline_pairs, div_pairs))
    proteins_agreement['divider'].append(compute_agreement_score(baseline_pairs, div_pairs))
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, div_pairs)
    proteins_spearman['divider'].append(sp)
    proteins_kendall['divider'].append(kt)

    bucket_pairs = set(bucket_mapping_proteins[idx])
    bucket_sum = sum(matrix[r, c] for r, c in bucket_pairs)
    proteins_min_sums['bucket'].append(bucket_sum)
    p, r, f = compute_metrics(baseline_pairs, bucket_pairs)
    proteins_precision['bucket'].append(p)
    proteins_recall['bucket'].append(r)
    proteins_f1['bucket'].append(f)
    proteins_gap['bucket'].append(compute_agreement_score(baseline_pairs, bucket_pairs))
    proteins_agreement['bucket'].append(compute_agreement_score(baseline_pairs, bucket_pairs))
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, bucket_pairs)
    proteins_spearman['bucket'].append(sp)
    proteins_kendall['bucket'].append(kt)

    direct_assignment = direct_mapping_proteins[idx]
    direct_pairs = set(direct_assignment.items())
    direct_sum = sum(matrix[r, c] for r, c in direct_pairs)
    proteins_min_sums['direct'].append(direct_sum)
    p, r, f = compute_metrics(baseline_pairs, direct_pairs)
    proteins_precision['direct'].append(p)
    proteins_recall['direct'].append(r)
    proteins_f1['direct'].append(f)
    proteins_gap['direct'].append(compute_agreement_score(baseline_pairs, direct_pairs))
    proteins_agreement['direct'].append(compute_agreement_score(baseline_pairs, direct_pairs))
    sp, kt = compute_rank_correlation(matrix, baseline_pairs, direct_pairs)
    proteins_spearman['direct'].append(sp)
    proteins_kendall['direct'].append(kt)


def plot_boxplot(metric_dict, metric_name, dataset_label):
    data = []
    for algo in ['divider', 'bucket', 'direct']:
        for val in metric_dict[algo]:
            data.append({'Algorithm': algo, metric_name: val})

    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Algorithm', y=metric_name, data=df, palette=colors)
    plt.title(f"{metric_name} Distribution - {dataset_label} Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save CSV
    csv_path = f'./csv/{timestamp}_{dataset_label}_{metric_name.lower().replace(" ", "_")}_boxplot.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved boxplot data CSV: {csv_path}")

    # Save plot
    filename = f'./png/{timestamp}_{dataset_label}_{metric_name.lower().replace(" ", "_")}_boxplot.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved boxplot PNG: {filename}")
    plt.show()


def plot_metric(metric_dict, metric_name, dataset_label):
    algorithms = ['divider', 'bucket', 'direct']
    means = [np.mean(metric_dict[algo]) for algo in algorithms]
    stds = [np.std(metric_dict[algo]) for algo in algorithms]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(algorithms, means, yerr=stds, capsize=5,
                   color=[colors['divider'], colors['bucket'], colors['direct']])
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison - {dataset_label} Dataset")
    plt.ylim(0, 110 if 'score' in metric_name.lower() else None)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot
    filename = f'./png/{timestamp}_{dataset_label}_{metric_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot PNG: {filename}")
    plt.show()


def create_summary_df(min_sums, precision, recall, f1, label, gap, agreement,
                      spearman=None, kendall=None, r2_dict=None):
    data = []
    for algo in ['lsa', 'divider', 'bucket', 'direct']:
        if algo == 'lsa':
            data.append([algo,
                         f"{np.mean(min_sums[algo]):.3f} ± {np.std(min_sums[algo]):.3f}",
                         '-', '-', '-', '-', '-', '-', '-', '-'])
        else:
            data.append([
                algo,
                f"{np.mean(min_sums[algo]):.3f} ± {np.std(min_sums[algo]):.3f}",
                f"{np.mean(precision[algo]):.1f}% ± {np.std(precision[algo]):.1f}%",
                f"{np.mean(recall[algo]):.1f}% ± {np.std(recall[algo]):.1f}%",
                f"{np.mean(f1[algo]):.1f}% ± {np.std(f1[algo]):.1f}%",
                f"{np.mean(gap[algo]):.2f}% ± {np.std(gap[algo]):.2f}%",
                f"{np.mean(agreement[algo]):.1f}% ± {np.std(agreement[algo]):.1f}%",
                f"{np.mean(spearman[algo]):.1f}% ± {np.std(spearman[algo]):.1f}%",
                f"{np.mean(kendall[algo]):.1f}% ± {np.std(kendall[algo]):.1f}%",
                f"{r2_dict[algo]:.2f}%"
            ])
    df = pd.DataFrame(data, columns=[
        'Algorithm', 'Mean Cost Sum', 'Precision', 'Recall', 'F1 Score',
        'Optimality Gap (%)', 'Agreement Score (%)',
        'Spearman (%)', 'Kendall (%)', 'R² (%)'
    ])
    csv_path = f'./csv/{timestamp}_{label}_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")
    return df


def save_per_instance_variability(label, precision, recall, f1, gap, agreement):
    records = []
    for algo in ['divider', 'bucket', 'direct']:
        for metric_name, values in [
            ('Precision', precision[algo]),
            ('Recall', recall[algo]),
            ('F1 Score', f1[algo]),
            ('Optimality Gap (%)', gap[algo]),
            ('Agreement Score (%)', agreement[algo])
        ]:
            records.append({
                'Algorithm': algo,
                'Metric': metric_name,
                'Min': np.min(values),
                '25%': np.percentile(values, 25),
                'Median': np.median(values),
                '75%': np.percentile(values, 75),
                'Max': np.max(values)
            })
    df = pd.DataFrame(records)
    csv_path = f'./csv/{timestamp}_{label}_variability.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved per-instance variability CSV: {csv_path}")


from scipy.stats import f_oneway, kruskal


def statistical_tests(metric_dict, metric_name, dataset_label):
    data = [metric_dict[algo] for algo in ['divider', 'bucket', 'direct']]
    anova_p = f_oneway(*data).pvalue
    kruskal_p = kruskal(*data).pvalue
    print(f"{dataset_label} - {metric_name} | ANOVA p={anova_p:.4e}, Kruskal-Wallis p={kruskal_p:.4e}")


mutag_r2 = compute_r2_for_min_sums(mutag_min_sums)
ohsu_r2 = compute_r2_for_min_sums(ohsu_min_sums)
proteins_r2 = compute_r2_for_min_sums(proteins_min_sums)

# Create and save summary tables (also prints)
mutag_summary_df = create_summary_df(mutag_min_sums, mutag_precision, mutag_recall, mutag_f1,
                                     "MUTAG", mutag_gap, mutag_agreement, mutag_spearman, mutag_kendall, mutag_r2)
ohsu_summary_df = create_summary_df(ohsu_min_sums, ohsu_precision, ohsu_recall, ohsu_f1,
                                    "OHSU", ohsu_gap, ohsu_agreement, ohsu_spearman, ohsu_kendall, ohsu_r2)
proteins_summary_df = create_summary_df(proteins_min_sums, proteins_precision, proteins_recall, proteins_f1, "PROTEINS",
                                        proteins_gap, proteins_agreement, proteins_spearman, proteins_kendall,
                                        proteins_r2)

# Save bar plots for MUTAG
plot_metric(mutag_precision, "Precision (%)", "MUTAG")
plot_metric(mutag_recall, "Recall (%)", "MUTAG")
plot_metric(mutag_f1, "F1 Score (%)", "MUTAG")

# Save bar plots for OHSU
plot_metric(ohsu_precision, "Precision (%)", "OHSU")
plot_metric(ohsu_recall, "Recall (%)", "OHSU")
plot_metric(ohsu_f1, "F1 Score (%)", "OHSU")

# Save bar plots for PROTEINS
plot_metric(proteins_precision, "Precision (%)", "PROTEINS")
plot_metric(proteins_recall, "Recall (%)", "PROTEINS")
plot_metric(proteins_f1, "F1 Score (%)", "PROTEINS")

# Save boxplots for MUTAG
plot_boxplot(mutag_f1, "F1 Score (%)", "MUTAG")

# Save boxplots for OHSU
plot_boxplot(ohsu_f1, "F1 Score (%)", "OHSU")

# Save boxplots for PROTEINS
plot_boxplot(proteins_f1, "F1 Score (%)", "PROTEINS")

save_per_instance_variability("MUTAG", mutag_precision, mutag_recall, mutag_f1, mutag_gap, mutag_agreement)
save_per_instance_variability("OHSU", ohsu_precision, ohsu_recall, ohsu_f1, ohsu_gap, ohsu_agreement)
save_per_instance_variability("PROTEINS", proteins_precision, proteins_recall, proteins_f1, proteins_gap,
                              proteins_agreement)
