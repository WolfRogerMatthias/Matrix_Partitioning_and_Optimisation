import matplotlib.pyplot as plt
from src.data.MockDataGenerator import MockDataGenerator
from src.optimize.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.greedy.GreedyAlgo import GreedyAlgo
import timeit

if __name__ == '__main__':
    MockDataGenerator = MockDataGenerator()
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    GreedyAlgo = GreedyAlgo(OptimizeAlgoApplied)

    matrix_sizes = [10, 20, 40, 60, 80, 100]
    number_of_matrices = 10000

    execution_times_linear = []
    execution_times_greedy = []

    for matrix_size in matrix_sizes:
        cost_matrices = MockDataGenerator.loadig_mock_data(f'mock_data_{matrix_size}.h5', number_of_matrices)
        execution_time_linear = timeit.timeit('[OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in range(number_of_matrices)]', globals=globals(), number=1)
        execution_times_linear.append(execution_time_linear)
        execution_time_greedy = timeit.timeit('GreedyAlgo.greedy_linear_applied(cost_matrices, 10000)', globals=globals(), number=1)
        execution_times_greedy.append(execution_time_greedy)

    print(matrix_sizes)
    print(execution_times_linear)
    print(execution_times_greedy)

    plt.plot(matrix_sizes, execution_times_linear, color='red')
    plt.plot(matrix_sizes, execution_times_greedy, color='blue')
    plt.xlim(min(matrix_sizes), max(matrix_sizes))
    #plt.ylim(min(execution_times_linear + execution_times_greedy), max(execution_times_linear + execution_times_greedy))
    plt.savefig(f'mock_data_together.png', format='png')
    plt.show()
