import numpy as np
import h5py
import timeit
import time
from src.optimize.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.greedy.GreedyAlgo import GreedyAlgo
from itertools import chain


class MockDataGenerator:

    def creating_mock_data(self, filename, num_of_matrices, matrix_size):
        with h5py.File(filename, 'w') as file:
            for i in range(num_of_matrices):
                matrix = np.random.uniform(0, 20, size=(matrix_size, matrix_size))
                file.create_dataset(f'cost_matrix_{i}', data=matrix)
            file.close()

    def loadig_mock_data(self, filename, num_of_matrices):
        with h5py.File(filename, 'r') as file:
            matrices = {i: np.array(file[f'cost_matrix_{i}']) for i in range(num_of_matrices)}
            file.close()
        return matrices


if __name__ == '__main__':
    MockDataGenerator = MockDataGenerator()
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    greedy_algo = GreedyAlgo()

#    MockDataGenerator.creating_mock_data('mock_data_60.h5', 10000, 60)

    cost_matrices = MockDataGenerator.loadig_mock_data('mock_data_60.h5', 10000)
