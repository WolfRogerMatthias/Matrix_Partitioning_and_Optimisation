
import numpy as np
from sklearn.cluster import KMeans
from src.OptimizeAlgoApplied import OptimizeAlgoApplied

class BucketAlgo:
    def __init__(self, OptimizeAlgoApplied):
        self.OptimizeAlgoApplied = OptimizeAlgoApplied

    def bucket_sub_matrices(self, matrix, number_of_buckets):
        rows, cols = matrix.shape
        min_arg = np.argmin(matrix, axis=1)
        bucket_with = cols // number_of_buckets
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
        while not correct_mapping and counter < 10:
            mapping.clear()
            counter += 1
            counts = []
            last_max = -1
            for key, (max, min, count) in my_dict.items():
                if (last_max > min):
                    min = last_max + 1
                if min != last_max + 1:
                    min = last_max + 1
                max = min + count - 1
                last_max = max
                my_dict[key] = (max, min, 0)
                counts.append(count)

            for element in min_arg:
                for key, (max, min, count) in my_dict.items():
                    if (min <= element <= max):
                        my_dict[key] = (max, min, count + 1)
                        mapping.append(key)
                        break

            correct_mapping = True

            for key, (max, min, count) in my_dict.items():
                if count != len(range(min, max + 1)) and correct_mapping:
                    correct_mapping = False

        print(mapping)
        print(min_arg)
        print(my_dict)
        sub_matrices = []
        mapping = np.array(mapping)
        for key, (max, min, count) in my_dict.items():
            if (count == 0):
                continue
            print(key)
            print(np.where(mapping == key)[0])
            row_ind = np.where(mapping == key)[0]
            sub_matrix = matrix[row_ind, :][:, min:max + 1]
            sub_matrices.append(sub_matrix)

        return sub_matrices




if __name__ == '__main__':
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    BucketAlgo = BucketAlgo(OptimizeAlgoApplied)

    matrix = np.random.uniform(0, 20, size=(20, 20))
    number_of_buckets = 4

    for row in matrix:
        for element in row:
            print(f'{element:6.2f} ', end='')
        print()

    sub_matrices = BucketAlgo.bucket_sub_matrices(matrix, number_of_buckets)
    





