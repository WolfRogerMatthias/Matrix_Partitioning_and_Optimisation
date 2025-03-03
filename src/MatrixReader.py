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
        row, col = self.matrix[pos].shape
        print(self.matrix[pos].shape)
        outputstr = ''
        for i in range(row):
            for j in range(col):
                outputstr += str('{:.3f}'.format(round(self.matrix[pos][i, j], 3))) + ' '
            outputstr += '\n'
        print(outputstr)

    def print_sub_matrix(self, matrix):
        if (type(matrix) == str):
            print(matrix)
            return 0
        else:
            row, col = matrix.shape
            print(matrix.shape)
            outputstr = ''
            for i in range(row):
                for j in range(col):
                    outputstr += str('{:.3f}'.format(round(matrix[i, j], 3))) + ' '
                outputstr += '\n'
            print(outputstr)

