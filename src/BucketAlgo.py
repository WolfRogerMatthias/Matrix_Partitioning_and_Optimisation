import numpy as np
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from itertools import chain
from src.MockDataGenerator import MockDataGenerator

class BucketAlgo:
    def __init__(self, OptimizeAlgoApplied):
        self.OptimizeAlgoApplied = OptimizeAlgoApplied

    def bucket_sub_matrices(self, matrix, number_of_buckets):
        rows, cols = matrix.shape
        min_arg = np.argmin(matrix, axis=1)
        bucket_with = cols / number_of_buckets
        my_dict = {i: (0, np.inf, 0) for i in range(number_of_buckets)}
        for element in min_arg:
            index = element // bucket_with
            if my_dict[index][0] <= element:
                my_dict[index] = (element, my_dict[index][1], my_dict[index][2])
            if my_dict[index][1] >= element:
                my_dict[index] = (my_dict[index][0], element, my_dict[index][2])
            my_dict[index] = (my_dict[index][0], my_dict[index][1], my_dict[index][2] + 1)

        correct_mapping = False
        counter = 0
        mapping = []
        while not correct_mapping:
            mapping = []
            counter += 1
            counts = []
            last_max = -1
            for key, (max, min, count) in my_dict.items():
                if (last_max > min):
                    min = last_max + 1
                if min != last_max + 1:
                    min = last_max + 1
                if key != number_of_buckets - 1:
                    max = min + count - 1
                else:
                    max = cols - 1
                last_max = max
                my_dict[key] = (max, min, 0)
                counts.append(count)

            for element in min_arg:
                for key, (max, min, count) in my_dict.items():
                    if min <= element <= max:
                        my_dict[key] = (max, min, count + 1)
                        mapping.append(key)
                        break

            correct_mapping = all(count <= (max_val - min_val + 1) for max_val, min_val, count in my_dict.values())


        sub_matrices = []
        mapping = np.array(mapping)
        applied_mapping = []
        for key, (max, min, count) in my_dict.items():
            if (count == 0):
                continue
            row_ind = np.where(mapping == key)[0]
            sub_matrix = matrix[row_ind, :][:, min:max + 1]
            applied_mapping.append(row_ind)
            sub_matrices.append(sub_matrix)

        return sub_matrices, applied_mapping

    def applied_mapping(self, matrices, numb_of_matrices, number_of_buckets):
        total_mapping = []
        for i in range(numb_of_matrices):

            sub_matrices, mapping = self.bucket_sub_matrices(matrices[i], number_of_buckets)
            row_mapping = []
            col_mapping = []
            len_mappings = 0
            for j in range(len(sub_matrices)):
                row_ind, col_ind = self.OptimizeAlgoApplied.compute_linear_sum_assignment(sub_matrices[j])
                if j >= 1:
                    col_mapping.append(col_ind + len_mappings)
                else:
                    col_mapping.append(col_ind)
                len_mappings += len(mapping[j])
                row_mapping.append(mapping[j])

            total_mapping.append(sorted(list(zip(list(chain(*row_mapping)), list(chain(*col_mapping)))), key=lambda x: x[0]))

        return total_mapping



