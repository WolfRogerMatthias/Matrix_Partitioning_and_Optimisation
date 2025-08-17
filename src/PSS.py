"""
This is for getting a subset of matrices of the Protiens dataset.
"""
import datetime
import time
import random
import numpy as np
import h5py


from itertools import combinations
from src.MockDataGenerator import MockDataGenerator

MockDataGenerator = MockDataGenerator()

len_proteins = 1113

number_of_matrices_proteins = len(list(combinations(range(len_proteins), r=2)))
start = time.time()
cost_matrices_proteins = MockDataGenerator.load_h5_file('./data/cost_matrices_PROTEINS.h5', number_of_matrices_proteins)
end = time.time()

print(f"Loaded {len(cost_matrices_proteins)} matrices in {end - start:.2f}s")

avg_rows_full = np.mean([(m.shape[0] + m.shape[1]) / 2 for m in cost_matrices_proteins.values()])
print(f"Average nodes (rows) in full dataset: {avg_rows_full}")

# For a subset
def get_representative_subset_by_rows(matrices_dict, subset_size, tolerance=0.005, max_attempts=1000):
    keys = list(matrices_dict.keys())  # preserve mapping
    n = len(keys)

    # Average number of rows for the full dataset
    full_avg_rows = np.mean([(mat.shape[0] + mat.shape[1]) / 2 for mat in matrices_dict.values()])

    for attempt in range(max_attempts):
        chosen_keys = random.sample(keys, subset_size)  # sample actual dict keys
        subset_avg_rows = np.mean([(matrices_dict[k].shape[0] + matrices_dict[k].shape[1]) / 2 for k in chosen_keys])

        if abs(subset_avg_rows - full_avg_rows) / full_avg_rows <= tolerance:
            print(f"Found representative subset on attempt {attempt + 1}")
            return chosen_keys, subset_avg_rows

    raise ValueError("Could not find a representative subset within tolerance.")


# Get subset
subset_indices, subset_mean = get_representative_subset_by_rows(
    cost_matrices_proteins,
    subset_size=10000,
    tolerance=0.005
)
print(f"Subset mean: {subset_mean}, difference: {abs(subset_mean - avg_rows_full)}")

# Extract matrices
subset_of_matrices = [cost_matrices_proteins[i] for i in subset_indices]

subset_file_path = './data/cost_matrices_PROTEINS_subset.h5'
with h5py.File(subset_file_path, 'w') as hf:
    # Save each matrix individually
    for i, mat in enumerate(subset_of_matrices):
        hf.create_dataset(f'cost_matrix_{i}', data=mat, dtype=np.float32)

    # Save the original indices for reference
    hf.create_dataset('indices', data=np.array(subset_indices, dtype=np.int64))


print(f"Saved subset of {len(subset_of_matrices)} matrices to {subset_file_path}")
