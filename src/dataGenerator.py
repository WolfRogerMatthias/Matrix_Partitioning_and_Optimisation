from src.MockDataGenerator import MockDataGenerator

"""
Creating my Mock data set for the timing analyzation and the accuracy
The total size of these creation is approximate 40GB
"""

MockDataGenerator = MockDataGenerator()

matrix_sizes = [i for i in range(10, 126, 5)]
number_of_matrices_ = [100, 1000, 10000, 25000]

for number_of_matrices in number_of_matrices_:
    for matrix_size in matrix_sizes:
        MockDataGenerator.creating_mock_data(f'./data/cost_matrices_{number_of_matrices}/cost_matrices_{matrix_size}.h5', number_of_matrices, matrix_size)

