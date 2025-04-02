import matplotlib.pyplot as plt
from src.data.MockDataGenerator import MockDataGenerator
from src.optimize.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.greedy.GreedyAlgo import GreedyAlgo
import timeit
import time
from itertools import chain

if __name__ == '__main__':
    MockDataGenerator = MockDataGenerator()
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    GreedyAlgo = GreedyAlgo()

    matrix_sizes = [10, 20, 40, 60, 80, 100]
    number_of_matrices = 10000

    execution_times_linear = []
    execution_times_greedy = []

    for matrix_size in matrix_sizes:
        cost_matrices = MockDataGenerator.loadig_mock_data(f'mock_data_{matrix_size}.h5', number_of_matrices)
        execution_time_linear = timeit.timeit('[OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in range(number_of_matrices)]', globals=globals(), number=1)
        execution_times_linear.append(execution_time_linear)
        start = time.time()
        total_mapping = []
        for i in range(len(cost_matrices)):
            rows = len(cost_matrices[i])
            cols = len(cost_matrices[i][0])
            row_interval = [round(rows / 4), round(rows / 2), round(rows / 4) * 3]
            col_interval = [round(cols / 4), round(cols / 2), round(cols / 4) * 3]
            sub_matrices = GreedyAlgo.greedy_sub_matrices(cost_matrices[i], row_interval, col_interval)

            mapping_row = []
            mapping_col = []
            for j in range(len(sub_matrices)):
                row, col = OptimizeAlgoApplied.compute_linear_sum_assignment(sub_matrices[j])
                mapping_row.append(row)
                mapping_col.append(col)

            complete_row_mapping = []
            complete_col_mapping = []
            for j in range(len(mapping_row)):
                if (j < 1):
                    complete_row_mapping.append(mapping_row[j])
                    complete_col_mapping.append(mapping_col[j])
                else:
                    col_len = col_interval[j - 1]
                    row_len = row_interval[j - 1]
                    complete_row_mapping.append((mapping_row[j] + row_len))
                    complete_col_mapping.append((mapping_col[j] + col_len))
            total_mapping.append([list(chain(*complete_row_mapping)), list(chain(*complete_col_mapping))])
        end = time.time()
        execution_times_greedy.append(end - start)

    plt.plot(matrix_sizes, execution_times_linear, color='red')
    plt.plot(matrix_sizes, execution_times_greedy, color='blue')
    plt.savefig(f'mock_data_together.png', format='png')
    plt.show()