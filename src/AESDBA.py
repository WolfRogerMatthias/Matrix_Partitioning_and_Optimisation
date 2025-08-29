import time
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
colors = {
    'lsa': '#333333',  # Black
    'divider': '#FF8C00',  # Orange
    'bucket': '#800080',  # Purple
    'direct': '#0000FF'  # Blue
}


matrix_sizes = [i for i in range(200, 401, 20)]
number_of_matrices = 1000

matrices = []
lsa_mappings = []
bucket_mappings = []

for matrix_size in matrix_sizes:
    start = time.time()
    cost_matrices = MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                                                   number_of_matrices)
    matrices.append(cost_matrices)
    lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[j]) for j in
                   range(number_of_matrices)]
    lsa_mappings.append(lsa_mapping)

    bucket_mapping = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
    bucket_mappings.append(bucket_mapping)

    end = time.time()
    print(f'{matrix_size:6}: {end - start:6.2f}')

"""
TODO i want a evalutaion i cann explian and are good plot/tables i can use.
RMSE
TP (F1)
R^2
scatter plots of diffrent matrix sizes
what about this fuck

"""


def evaluation_metrics(cost_matrices, optimal_mappings, approx_mappings):
    tp = []
    p_gap = []
    p_node_gap = []
    opt_sums = []
    app_sums = []
    optimal_elements = []
    approx_elements = []
    opt_mappings = []
    app_mappings = []
    for i in range(len(cost_matrices)):
        matrice = cost_matrices[i]
        optimal_mapping = optimal_mappings[i]
        approx_mapping = approx_mappings[i]
        ac_rate = []
        # Assignment Accuracy
        gap_rate = []
        node_gap_rate = []
        opt_s = []
        app_s = []
        opt_elements = []
        app_elements = []
        opt_mapping = []
        app_mapping = []
        for j in range(len(optimal_mapping)):
            count = 0
            y_opt_map = optimal_mapping[j][0]
            x_opt_map = optimal_mapping[j][1]
            y_app_map = approx_mapping[j]
            x_app_map = approx_mapping[j]
            matrix = matrice[j]
            opt_ele = matrix[y_opt_map, x_opt_map]
            app_ele = [matrix[x, y] for x, y in approx_mapping[j]]
            opt_sum = sum(opt_ele)
            app_sum = sum(app_ele)
            gap = (app_sum - opt_sum) / opt_sum
            n = len(y_opt_map)
            n_gap_r = []

            for k in range(n):
                count += 1 if (y_opt_map[k] == y_app_map[k][0] and x_opt_map[k] == x_app_map[k][1]) else 0
                node_gap = (app_ele[k] - opt_ele[k]) / opt_ele[k]
                n_gap_r.append(node_gap)

            ac_rate.append(count / n)
            gap_rate.append(gap)
            node_gap_rate.append(n_gap_r)
            opt_s.append(opt_sum)
            app_s.append(app_sum)
            opt_elements.append(opt_ele)
            app_elements.append(app_ele)
            opt_mapping.append(optimal_mapping[j])
            app_mapping.append(approx_mapping[j])

        tp.append(ac_rate)
        p_gap.append(gap_rate)
        p_node_gap.append(node_gap_rate)
        opt_sums.append(opt_s)
        app_sums.append(app_s)
        optimal_elements.append(opt_elements)
        approx_elements.append(app_elements)
        opt_mappings.append(opt_mapping)
        app_mappings.append(approx_mapping)

    data = {
        'tp': np.array(tp),
        'cost_gap': np.array(p_gap),
        'node_wise_cost_gap': p_node_gap,
        'opt_sum': np.array(opt_sums),
        'app_sum': np.array(app_sums),
        'optimal_elements': optimal_elements,
        'approx_elements': approx_elements,
        'optimal_mappings': opt_mappings,
        'approx_mappings': approx_mappings,
    }

    return data


data = evaluation_metrics(matrices, lsa_mappings, bucket_mappings)


# --- Assignment Accuracy ---
mean_acc = np.mean(data['tp'], axis=1)
std_acc = np.std(data['tp'], axis=1)

plt.figure(figsize=(8, 6))
plt.plot(matrix_sizes, mean_acc, label='Mean Assignment Accuracy', color=colors['bucket'])
plt.fill_between(matrix_sizes, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color=colors['bucket'])
plt.xlabel('Matrix Size')
plt.ylabel('Assignment Accuracy')
plt.xticks(matrix_sizes)
plt.ylim(0, 1)
plt.xlim(matrix_sizes[0], matrix_sizes[-1])
plt.title('Assignment Accuracy of Bucket Algorithm')
plt.grid(True)
plt.legend()
plt.savefig(f'./png/{timestamp}_acc_bucket.png')
plt.show()


# --- Cost Gap ---
mean_gap = np.mean(data['cost_gap'], axis=1)
std_gap = np.std(data['cost_gap'], axis=1)

plt.figure(figsize=(8, 6))
plt.plot(matrix_sizes, mean_gap, label='Mean Cost Gap', color=colors['bucket'])
plt.fill_between(matrix_sizes, mean_gap - std_gap, mean_gap + std_gap, alpha=0.2, color=colors['bucket'])
plt.xlabel('Matrix Size')
plt.ylabel('Cost Gap')
plt.xticks(matrix_sizes)
plt.ylim(0, None)
plt.xlim(matrix_sizes[0], matrix_sizes[-1])
plt.title('Cost Gap of Bucket Algorithm')
plt.grid(True)
plt.legend()
plt.savefig(f'./png/{timestamp}_cost_gap_bucket.png')
plt.show()
