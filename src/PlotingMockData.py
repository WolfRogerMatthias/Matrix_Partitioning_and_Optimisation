import matplotlib.pyplot as plt
from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.GreedyAlgo import GreedyAlgo
from src.GreedyAlgoExtended import GreedyAlgoExtended
from src.GreedyAlgoDynamic import GreedyAlgoDynamic
from src.GreedyAlgoDynamicExtended import GreedyAlgoDynamicExtended
from src.BucketAlgo import BucketAlgo
from src.GreedyAlgoExtendedSelf import GreedyAlgoExtendedSelf
import numpy as np
import os
import timeit
import time
import h5py

if __name__ == '__main__':
    MockDataGenerator = MockDataGenerator()
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    GreedyAlgo = GreedyAlgo(OptimizeAlgoApplied)
    GreedyAlgoExtended = GreedyAlgoExtended(OptimizeAlgoApplied)
    GreedyAlgoDynamic = GreedyAlgoDynamic(OptimizeAlgoApplied)
    GreedyAlgoDynamicExtended = GreedyAlgoDynamicExtended(OptimizeAlgoApplied)
    BucketAlgo = BucketAlgo(OptimizeAlgoApplied)
    GreedyAlgoExtendedSelf = GreedyAlgoExtendedSelf(OptimizeAlgoApplied)

    """
    matrix_sizes = [i for i in range(10, 91, 5)]
    num_of_matrices = 25000

    execution_times_linear = []
    execution_times_greedy = []
    execution_times_greedy_extended = []

    print(matrix_sizes)
    start = time.time()
    for matrix_size in matrix_sizes:
        filepath = f'./data/mock_data_{matrix_size}.h5'
        MockDataGenerator.creating_mock_data(filepath, num_of_matrices, matrix_size)
        cost_matrices = MockDataGenerator.loadig_mock_data(filepath, num_of_matrices)
        execution_time_linear = timeit.timeit('[OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in range(num_of_matrices)]', globals=globals(), number=1)
        execution_times_linear.append(execution_time_linear)
        execution_time_greedy = timeit.timeit('GreedyAlgo.greedy_linear_applied(cost_matrices, num_of_matrices)', globals=globals(), number=1)
        execution_times_greedy.append(execution_time_greedy)
        #execution_time_greedy_extended = timeit.timeit('GreedyAlgoExtended.greedy_linear_applied(cost_matrices, num_of_matrices)', globals=globals(), number=1)
        #execution_times_greedy_extended.append(execution_time_greedy_extended)
        os.remove(filepath)
    end = time.time()
    print(f'Execution time: {end - start} seconds')
    plt.plot(matrix_sizes, execution_times_linear, color='red')
    plt.plot(matrix_sizes, execution_times_greedy, color='blue')
    #plt.plot(matrix_sizes, execution_times_greedy_extended, color='green')
    plt.xlim(min(matrix_sizes), max(matrix_sizes))
    plt.xticks(matrix_sizes, rotation=90)
    plt.title(f'For number of matrices {num_of_matrices}')
    plt.xlabel('Matrix size')
    plt.legend(['Linear Sum Assignment', 'Greedy Assignment'])
    plt.savefig(f'./png/mock_data_linear_vs_greedy_{num_of_matrices}_{matrix_sizes[-1]}.png', format='png')
    plt.show()

    with h5py.File(f'./data/timings_greedy_{num_of_matrices}_{matrix_sizes[-1]}.h5', 'w') as file:
        file.create_dataset('linear_sum_assignment_timing', data=execution_times_linear)
        file.create_dataset('greedy_sum_assignment_timing', data=execution_times_greedy)
        #file.create_dataset('greedy_extended_sum_assignment_timing', data=execution_times_greedy_extended)
        file.create_dataset('matrix_sizes', data=matrix_sizes)
        file.close()

    """
    """    
    matrix_sizes = [i for i in range(10, 121, 10)]
    num_of_matrices = 25000
    execution_times_linear = []
    execution_times_greedy = []
    execution_times_greedy_dynamic = []
    execution_times_greedy_extended = []
    execution_times_greedy_dynamic_extended = []


    print(matrix_sizes)
    start = time.time()
    for matrix_size in matrix_sizes:
        print(f'Run of matrix_size {matrix_size}')
        filepath = f'./data/mock_data_{matrix_size}.h5'
        MockDataGenerator.creating_mock_data(filepath, num_of_matrices, matrix_size)
        cost_matrices = MockDataGenerator.loadig_mock_data(filepath, num_of_matrices)
        execution_time_linear = timeit.timeit('[OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in range(num_of_matrices)]', globals=globals(), number=1)
        execution_times_linear.append(execution_time_linear)
        execution_time_greedy_dynamic = timeit.timeit('GreedyAlgoDynamic.greedy_linear_applied(cost_matrices, num_of_matrices)', globals=globals(), number=1)
        execution_times_greedy_dynamic.append(execution_time_greedy_dynamic)
        execution_time_greedy = timeit.timeit('GreedyAlgo.greedy_linear_applied(cost_matrices, num_of_matrices)', globals=globals(), number=1)
        execution_times_greedy.append(execution_time_greedy)
        execution_time_greedy_extended = timeit.timeit('GreedyAlgoExtended.greedy_linear_applied(cost_matrices, num_of_matrices)', globals=globals(), number=1)
        execution_times_greedy_extended.append(execution_time_greedy_extended)

        execution_time_greedy_dynamic_extended = timeit.timeit('GreedyAlgoDynamicExtended.greedy_linear_applied(cost_matrices, num_of_matrices, 38)', globals=globals(), number=1)
        execution_times_greedy_dynamic_extended.append(execution_time_greedy_dynamic_extended)
        os.remove(filepath)
    end = time.time()

    print(f'Execution time: {end - start} seconds')
    plt.plot(matrix_sizes, execution_times_linear, color='red')
    plt.plot(matrix_sizes, execution_times_greedy, color='blue')
    plt.plot(matrix_sizes, execution_times_greedy_dynamic, color='green')
    plt.plot(matrix_sizes, execution_times_greedy_extended, color='yellow')
    plt.plot(matrix_sizes, execution_times_greedy_dynamic_extended, color='m')
    plt.xlim(min(matrix_sizes), max(matrix_sizes))
    plt.xticks(matrix_sizes, rotation=90)
    plt.title(f'For number of matrices {num_of_matrices}')
    plt.xlabel('Matrix size')
    plt.legend(['Linear Sum Assignment', 'Greedy Assignment', 'Greedy Dynamic Assignment', 'Greedy Extended Assignment', 'Greedy Dynamic Extended Assignment'])
    plt.savefig(f'./png/dynamic_{num_of_matrices}_{matrix_sizes[-1]}.png', format='png')
    plt.show()

    """

    matrix_sizes = [i for i in range(10, 121, 5)]
    num_of_matrices = 10000

    execution_times_linear = []
    execution_times_bucket = []
    execution_times_greedy = []
    execution_times_greedy_extended = []
    execution_times_greedy_extended_self =[]


    print(matrix_sizes)

    start = time.time()
    for matrix_size in matrix_sizes:
        print(f'Run of matrix_size {matrix_size}')
        filepath = f'./data/mock_data_{matrix_size}.h5'
        MockDataGenerator.creating_mock_data(filepath, num_of_matrices, matrix_size)
        cost_matrices = MockDataGenerator.loadig_mock_data(filepath, num_of_matrices)
        execution_time_linear = timeit.timeit(
            '[OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in range(num_of_matrices)]',
            globals=globals(), number=1)
        execution_times_linear.append(execution_time_linear)
        """
        execution_time_bucket = timeit.timeit(
            'BucketAlgo.applied_mapping(cost_matrices, num_of_matrices)',
            globals=globals(), number=1
        )
        execution_times_bucket.append(execution_time_bucket)
        """
        execution_time_greedy = timeit.timeit('GreedyAlgo.greedy_linear_applied(cost_matrices, num_of_matrices)',
            globals=globals(), number=1
        )
        execution_times_greedy.append(execution_time_greedy)
        execution_time_greedy_extended = timeit.timeit(
            'GreedyAlgoExtended.greedy_linear_applied(cost_matrices, num_of_matrices)',
            globals=globals(), number=1
        )
        execution_times_greedy_extended.append(execution_time_greedy_extended)
        execution_time_greedy_extended_self = timeit.timeit(
            'GreedyAlgoExtendedSelf.greedy_linear_applied(cost_matrices, num_of_matrices)',
            globals=globals(), number=1
        )
        execution_times_greedy_extended_self.append(execution_time_greedy_extended_self)
        os.remove(
            filepath
        )
    end = time.time()
    print(f'Execution time: {end - start} seconds')
    plt.plot(matrix_sizes, execution_times_linear, color='red')
    #plt.plot(matrix_sizes, execution_times_bucket, color='blue')
    plt.plot(matrix_sizes, execution_times_greedy, color='green')
    plt.plot(matrix_sizes, execution_times_greedy_extended, color='yellow')
    plt.plot(matrix_sizes, execution_times_greedy_extended_self, color='m')
    plt.xlim(min(matrix_sizes), max(matrix_sizes))
    plt.xticks(matrix_sizes, rotation=90)
    plt.title(f'For number of matrices {num_of_matrices}')
    plt.xlabel('Matrix size')
    plt.legend(['LSA', 'Greedy', 'Extended', 'Extended Self'])
    plt.savefig(f'./png/linear_vs_greedy_vs_extended_vs_self_{num_of_matrices}_{matrix_sizes[-1] - matrix_sizes[-2]}.png', format='png')
    plt.show()






