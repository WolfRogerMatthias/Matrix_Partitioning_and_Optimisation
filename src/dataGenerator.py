from src.MockDataGenerator import MockDataGenerator
import time

MockDataGenerator = MockDataGenerator()

matrix_sizes = [i for i in range(245, 251, 5)]


for matrix_size in matrix_sizes:
    start = time.time()
    MockDataGenerator.creating_mock_data(
        f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
        1000, matrix_size)
    end = time.time()
    print(f'End of matrix size {matrix_size}. Time elapsed: {end - start}')



for matrix_size in matrix_sizes:
    start = time.time()
    filepath = f'./data/cost_matrices_asymmetric/cost_matrices_asymmetric_{matrix_size}.h5'
    MockDataGenerator.creating_asymmetric_mock_data(filepath, 1000, matrix_size)
    end = time.time()
    print(f'End of matrix size {matrix_size}. Time elapsed: {end - start}')

