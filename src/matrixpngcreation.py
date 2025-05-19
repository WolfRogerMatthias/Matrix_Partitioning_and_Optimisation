import matplotlib.pyplot as plt
import numpy as np
from src.MockDataGenerator import MockDataGenerator
import time

MockDataGenerator = MockDataGenerator()

matrix_sizes = [i for i in range(10, 121, 5)]


for matrix_size in matrix_sizes:
    print(f'Start matrix size {matrix_size}')
    start = time.time()
    MockDataGenerator.creating_mock_data(
        f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
        10000, matrix_size)
    end = time.time()
    print(f'End of matrix size {matrix_size}. Time elapsed: {end - start}')
