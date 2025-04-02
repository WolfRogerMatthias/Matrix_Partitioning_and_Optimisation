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

    start_time = time.time()
    total_mapping = []
    for i in range(len(cost_matrices)):
        rows = len(cost_matrices[i])
        cols = len(cost_matrices[i][0])
        row_interval = [round(rows / 4), round(rows / 2), round(rows / 4) * 3]
        col_interval = [round(cols / 4), round(cols / 2), round(cols / 4) * 3]
        sub_matrices = greedy_algo.greedy_sub_matrices(cost_matrices[i], row_interval, col_interval)

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
    end_time = time.time()

    start = time.time()
    mapping_new = []
    for i in range(10000):
        mapping = OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i])
        mapping_new.append(mapping)
    end = time.time()

    print(f'Greedy {end_time - start_time} seconds vs applied {end - start} seconds ')