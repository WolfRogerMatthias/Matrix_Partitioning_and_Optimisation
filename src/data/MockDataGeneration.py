import numpy as np
import h5py


def print_matrix(matrix):
    max_int_digits = max(len(str(int(num))) for row in matrix for num in row)

    format_str = f"{{:>{max_int_digits + 3}.2f}}"
    print(matrix.shape)
    for row in matrix:
        formatted_row = [format_str.format(num) for num in row]
        print(" ".join(formatted_row))
    print()

def creating_mock_data(filename='mock_data.h5', num_of_matrices=10000, matrix_size=10):
    with h5py.File(filename, 'w') as file:
        for i in range(num_of_matrices):
            matrix = np.random.uniform(0, 200, size=(matrix_size, matrix_size))
            file.create_dataset(f'cost_matrix_{matrix_size}_{i}', data=matrix)
        file.close()

def loading_mock_data(filename='mock_data.h5', num_of_matrices=10000, matrix_size=10):
    with h5py.File(filename, 'r') as file:
        matrices = {i: np.array(file[f'cost_matrix_{matrix_size}_{i}']) for i in range(num_of_matrices)}
        file.close()
    return matrices

creating_mock_data(filename='mock_data_10.h5', num_of_matrices=10000, matrix_size=10)
creating_mock_data(filename='mock_data_20.h5', num_of_matrices=10000, matrix_size=20)
creating_mock_data(filename='mock_data_40.h5', num_of_matrices=10000, matrix_size=40)

matrices_10 = loading_mock_data('mock_data_10.h5', num_of_matrices=10000, matrix_size=10)
matrices_20 = loading_mock_data('mock_data_20.h5', num_of_matrices=10000, matrix_size=20)
matrices_40 = loading_mock_data('mock_data_40.h5', num_of_matrices=10000, matrix_size=40)

print_matrix(matrices_10[0])
print_matrix(matrices_20[0])
print_matrix(matrices_40[0])

print_matrix(matrices_10[100])
print_matrix(matrices_20[100])
print_matrix(matrices_40[100])

print_matrix(matrices_10[9999])
print_matrix(matrices_20[9999])
print_matrix(matrices_40[9999])
