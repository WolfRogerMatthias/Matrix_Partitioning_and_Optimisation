import numpy as np
import h5py

class MockDataGenerator:

    def creating_mock_data(self, filepath, num_of_matrices, matrix_size):
        with h5py.File(filepath, 'w') as file:
            for i in range(num_of_matrices):
                matrix = np.random.uniform(0.1,10,size=(matrix_size, matrix_size))
                file.create_dataset(f'cost_matrix_{i}', data=matrix)
            file.close()

    def loadig_mock_data(self, filepath, num_of_matrices):
        with h5py.File(filepath, 'r') as file:
            matrices = {i: np.array(file[f'cost_matrix_{i}']) for i in range(num_of_matrices)}
            file.close()
        return matrices


if __name__ == '__main__':
    MockDataGenerator = MockDataGenerator()
