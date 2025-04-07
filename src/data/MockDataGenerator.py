import numpy as np
import h5py

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

    MockDataGenerator.creating_mock_data('mock_data_10.h5', 10000, 10)
    MockDataGenerator.creating_mock_data('mock_data_20.h5', 10000, 20)
    MockDataGenerator.creating_mock_data('mock_data_40.h5', 10000, 40)
    MockDataGenerator.creating_mock_data('mock_data_60.h5', 10000, 60)
    MockDataGenerator.creating_mock_data('mock_data_80.h5', 10000, 80)
    MockDataGenerator.creating_mock_data('mock_data_100.h5', 10000, 100)
    MockDataGenerator.creating_mock_data('mock_data_200.h5', 10000, 200)
    MockDataGenerator.creating_mock_data('mock_data_400.h5', 10000, 400)

