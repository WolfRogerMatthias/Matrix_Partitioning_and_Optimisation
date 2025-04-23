import matplotlib.pyplot as plt
import random
from src.MockDataGenerator import MockDataGenerator

"""
To see how the matrix are and to display some random ones
This is a check so that every mockdata matrix is create evenly
"""

MockDataGenerator = MockDataGenerator()

matrix_sizes = [i for i in range(10, 126, 5)]
number_of_matrices_ = [100, 1000, 10000, 25000]

cost_matrices = []

for number_of_matrices in number_of_matrices_:
    for matrix_size in matrix_sizes:
        cost = MockDataGenerator.loadig_mock_data(f'./data/cost_matrices_{number_of_matrices}/cost_matrices_{matrix_size}.h5', number_of_matrices)
        cost_matrices += [i for i in random.sample(list(cost.values()), 1)]

for i in range(len(cost_matrices)):
    plt.imshow(cost_matrices[i], cmap='RdYlBu_r', interpolation='nearest')
    plt.show()