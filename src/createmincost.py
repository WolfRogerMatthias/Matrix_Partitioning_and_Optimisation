import matplotlib.pyplot as plt

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.GreedyAlgo import GreedyAlgo
from src.GreedyAlgoExtended import GreedyAlgoExtended
from src.GreedyAlgoDynamicExtended import GreedyAlgoDynamicExtended
from src.GreedyAlgoDynamic import GreedyAlgoDynamic
from src.BucketAlgo import BucketAlgo
import time
import numpy as np
import datetime
"""
First approach for the creation of the minimal cost for linear and greedy algo
This is just to make the first approach to creating some metrics now need to adapt to all the 4 algos
"""

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
GreedyAlgo = GreedyAlgo(OptimizeAlgoApplied)
GreedyAlgoExtended = GreedyAlgoExtended(OptimizeAlgoApplied)
GreedyAlgoDynamicExtended = GreedyAlgoDynamicExtended(OptimizeAlgoApplied)
GreedyAlgoDynamic = GreedyAlgoDynamic(OptimizeAlgoApplied)
BucketAlgo = BucketAlgo(OptimizeAlgoApplied)


matrix_sizes = [i for i in range(700, 1101, 20)]
number_of_matrices_ = [1000]


linear_timings = []
greedy_timings = []
bucket_timings = []
min_cost_linear = []
min_cost_greedy = []
min_cost_bucket = []
for number_of_matrices in number_of_matrices_:
    start = time.time()
    for matrix_size in matrix_sizes:
        start_matrix_size = time.time()
        cost = MockDataGenerator.loadig_mock_data(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5', number_of_matrices)
        linear_start = time.time()
        mapping_linear = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost[i]) for i in range(number_of_matrices)]
        linear_end = time.time()
        greedy_start = time.time()
        mapping_greedy = GreedyAlgo.greedy_linear_applied(cost, number_of_matrices)
        greedy_end = time.time()
        bucket_start = time.time()
        mapping_bucket = BucketAlgo.applied_mapping(cost, number_of_matrices)
        bucket_end = time.time()
        linear_timings.append(linear_end - linear_start)
        greedy_timings.append(greedy_end - greedy_start)
        bucket_timings.append(bucket_end - bucket_start)
        min_linear = []
        min_greedy = []
        min_bucket = []
        for i in range(number_of_matrices):
            min_linear.append(cost[i][mapping_linear[i][0], mapping_linear[i][1]].sum())
            min_greedy.append(cost[i][mapping_greedy[i][0], mapping_greedy[i][1]].sum())
            min_bucket.append(sum(cost[i][x][y] for x, y in mapping_bucket[i]))
        min_cost_linear.append(min_linear)
        min_cost_greedy.append(min_greedy)
        min_cost_bucket.append(min_bucket)
        end_matrix_size = time.time()
        print(f'End of matrix size {matrix_size}. Time elapsed: {end_matrix_size - start_matrix_size:06.4f}, Linear elapsed: {linear_end - linear_start:06.4f}, Greedy elapsed: {greedy_end - greedy_start:06.4f}, Bucket elapsed: {bucket_end - bucket_start:06.4f}')

    end = time.time()
    print(f'End of number of matrices {number_of_matrices}. Time elapsed: {end - start}')

"""
mae = []
mse = []
mape = []
smape = []
for i in range(len(min_cost_linear)):
    mae.append(np.mean(np.abs(np.array(min_cost_linear[i]) - np.array(min_cost_greedy[i]))))
    mse.append(np.mean((np.array(min_cost_linear[i]) - np.array(min_cost_greedy[i])) ** 2))
    mape.append(np.mean(np.abs((np.array(min_cost_linear[i]) - np.array(min_cost_greedy[i])) / np.array(min_cost_linear[i]))) * 100)
    smape.append(np.mean(
        np.abs(
            np.array(min_cost_linear[i]) - np.array(min_cost_greedy[i])
        ) / (
                (np.abs(np.array(min_cost_linear[i])) + np.abs(np.array(min_cost_greedy[i]))) / 2 + 1e-10
        )
    ) * 100)
print(len(min_cost_linear), len(min_cost_greedy), len(mape), len(matrix_sizes))
"""
"""
for j in range(len(number_of_matrices_)):
    fig, axis = plt.subplots()
    for i in range(len(matrix_sizes)):
        axis.plot(matrix_sizes, linear_timings[len(matrix_sizes)*i:len(matrix_sizes)*(i + 1)], color='#0072B2')
        axis.plot(matrix_sizes, greedy_timings[len(matrix_sizes)*i:len(matrix_sizes)*(i + 1)], color='blue')
        axis.scatter([matrix_sizes[i]-2]*len(min_cost_linear[i]), min_cost_linear[i], color='red', s=5)
        axis.scatter([matrix_sizes[i]+2]*len(min_cost_greedy[i]), min_cost_greedy[i], color='blue', s=5)

    plt.legend(['Linear', 'Bucket'])
    plt.ylabel('Minimal Sum')
    plt.xlabel('Size of (n x n) matrix')
    plt.xticks(matrix_sizes, rotation=90)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.title(f'Iterate over {number_of_matrices_[j]} matrices')
    plt.tight_layout()
    plt.savefig(f'./png/presentation/{timestamp}.png')
    plt.show()
"""
for j in range(len(number_of_matrices_)):
    fig, axis = plt.subplots()
    positions_linear = [matrix_sizes[i] - 5 for i in range(len(matrix_sizes))]
    positions_greedy = [matrix_sizes[i] for i in range(len(matrix_sizes))]
    positions_bucket = [matrix_sizes[i] + 5 for i in range(len(matrix_sizes))]

    """
    # Boxplot for Greedy
    box_greedy = axis.boxplot(
        [min_cost_greedy[i] for i in range(len(matrix_sizes))],
        positions=positions_greedy,
        widths=10,
        patch_artist=True,
        boxprops=dict(facecolor='blue', color='blue'),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none')
    )
    """

    # Boxplot for Linear
    box_linear = axis.boxplot(
        [min_cost_linear[i] for i in range(len(matrix_sizes))],
        positions=positions_linear,
        widths=10,
        patch_artist=True,
        boxprops=dict(facecolor='red', color='red'),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none')
    )

    box_bucket = axis.boxplot(
        [min_cost_bucket[i] for i in range(len(matrix_sizes))],
        positions=positions_bucket,
        widths=10,
        patch_artist=True,
        boxprops=dict(facecolor='orange', color='orange'),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none')
    )

    plt.legend([box_linear["boxes"][0], box_bucket["boxes"][0]], ['Linear', 'Bucket'], loc='upper left')
    plt.ylabel('Minimal Sum')
    plt.xlabel('Size of (n x n) matrix')
    plt.xticks(matrix_sizes, labels=matrix_sizes, rotation=90)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.title(f'Iterate over {number_of_matrices_[j]} matrices')
    plt.tight_layout()
    plt.savefig(f'./png/boxplots/{timestamp}.png')
    plt.show()


for j in range(len(number_of_matrices_)):
    fig, axis = plt.subplots()

    axis.plot(matrix_sizes, linear_timings[len(matrix_sizes)*j:len(matrix_sizes)*(j + 1)], color='red')
    #axis.plot(matrix_sizes, greedy_timings[len(matrix_sizes)*j:len(matrix_sizes)*(j + 1)], color='blue')
    axis.plot(matrix_sizes, bucket_timings[len(matrix_sizes)*j:len(matrix_sizes)*(j + 1)], color='orange')

    plt.legend(['Linear', 'Bucket'])
    plt.ylabel('Time elapsed in s')
    plt.xlabel('Size of (n x n) matrix')
    plt.xticks(matrix_sizes, rotation=90)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.title(f'Iterate over {number_of_matrices_[j]} matrices')
    plt.tight_layout()
    plt.savefig(f'./png/results/{timestamp}.png')
    plt.show()


