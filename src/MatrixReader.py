import numpy as np
import h5py

class MatrixReader:
    def __init__(self):
        self.matrix = []

    def load_h5_file(self, filename, num_matrices):
        with h5py.File(filename, 'r') as ffile:
            self.matrix = {i: np.array(ffile[f'cost_matrix_{i}']) for i in range(num_matrices)}
        return self.matrix

    def print_matrix(self, pos):
        max_int_digits = max(len(str(int(num))) for row in self.matrix[pos] for num in row)

        format_str = f"{{:>{max_int_digits + 3}.2f}}"
        print(self.matrix[pos].shape)
        for row in self.matrix[pos]:
            formatted_row = [format_str.format(num) for num in row]
            print(" ".join(formatted_row))
        print()

    def print_sub_matrix(self, matrix):
        if (type(matrix) == str):
            print(matrix)
            return 0
        else:
            max_int_digits = max(len(str(int(num))) for row in matrix for num in row)

            format_str = f"{{:>{max_int_digits + 3}.2f}}"
            print(matrix.shape)
            for row in self.matrix:
                formatted_row = [format_str.format(num) for num in row]
                print(" ".join(formatted_row))
            print()

