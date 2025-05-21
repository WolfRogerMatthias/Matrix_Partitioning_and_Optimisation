import time
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.GreedyAlgo import GreedyAlgo
from src.BucketAlgo import BucketAlgo


MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
GreedyAlgo = GreedyAlgo(OptimizeAlgoApplied)
BucketAlgo = BucketAlgo(OptimizeAlgoApplied)

matrix_sizes = [i for i in range(700, 1101, 20)]
number_of_matrices = 250

greedy_true_positives = []
bucket_true_positives = []
start_complete = time.time()
for matrix_size in matrix_sizes:
    start = time.time()
    cost_matrix = MockDataGenerator.loadig_mock_data(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5', number_of_matrices)
    linear_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrix[i]) for i in range(number_of_matrices)]
    greedy_mapping = GreedyAlgo.greedy_linear_applied(cost_matrix, number_of_matrices)
    bucket_mapping = BucketAlgo.applied_mapping(cost_matrix, number_of_matrices)
    greedy_counts = []
    bucket_counts = []
    for i in range(number_of_matrices):
        greedy_count = 0
        bucket_count = 0
        for j in range(len(linear_mapping[i][0])):
            if linear_mapping[i][0][j] == greedy_mapping[i][0][j] and linear_mapping[i][1][j] == greedy_mapping[i][1][j]:
                greedy_count += 1
            if bucket_mapping[i][j][0] == linear_mapping[i][0][j] and bucket_mapping[i][j][1] == linear_mapping[i][1][j]:
                bucket_count += 1
        greedy_counts.append(greedy_count / matrix_size)
        bucket_counts.append(bucket_count / matrix_size)
    greedy_true_positives.append(greedy_counts)
    bucket_true_positives.append(bucket_counts)
    end = time.time()
    print(f'End of matrix size {matrix_size}. Time elapsed: {end - start}')
end_complete = time.time()
print(f'Total time elapsed: {end_complete - start_complete}')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig, axis = plt.subplots()
axis.boxplot(
    greedy_true_positives,
    positions=matrix_sizes,
    widths=5,
    patch_artist=True,
    boxprops=dict(facecolor='blue', color='blue'),
    medianprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none')
)
axis.set_ylim(-.025, 1.025)
axis.set_xticks(matrix_sizes)
axis.set_title("Greedy True Positives")
axis.set_xlabel("Matrix Size")
axis.set_ylabel("True Positive Rate")
plt.xticks(rotation=90)
plt.savefig(f'./png/truepos/{timestamp}_greedy.png')
plt.show()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig, axis = plt.subplots()
axis.boxplot(
    bucket_true_positives,
    positions=matrix_sizes,
    widths=5,
    patch_artist=True,
    boxprops=dict(facecolor='orange', color='orange'),
    medianprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none')
)
axis.set_ylim(-.025, 1.025)
axis.set_xticks(matrix_sizes)
axis.set_title("Bucket True Positives")
axis.set_xlabel("Matrix Size")
axis.set_ylabel("True Positive Rate")
plt.xticks(rotation=90)
plt.savefig(f'./png/truepos/{timestamp}_bucket.png')
plt.show()

greedy_data = []
for size, values in zip(matrix_sizes, greedy_true_positives):
    for v in values:
        greedy_data.append({'Matrix Size': size, 'True Positives': v})

df_greedy = pd.DataFrame(greedy_data)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(10, 6))
sns.violinplot(x="Matrix Size", y="True Positives", data=df_greedy, palette='Blues', hue='Matrix Size', legend=False)
plt.ylim(None,1)
plt.title('True Positives by Matrix Size')
plt.savefig(f'./png/truepos/violin/{timestamp}_greedy.png')
plt.show()

bucket_data = []
for size, values in zip(matrix_sizes, bucket_true_positives):
    for v in values:
        bucket_data.append({'Matrix Size': size, 'True Positives': v})

df_bucket = pd.DataFrame(bucket_data)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(10, 6))
sns.violinplot(x="Matrix Size", y="True Positives", data=df_bucket, palette='Oranges', hue='Matrix Size', legend=False)
plt.ylim(-.025, 1.025)
plt.title('True Positives by Matrix Size')
plt.savefig(f'./png/truepos/violin/{timestamp}_bucket.png')
plt.show()
