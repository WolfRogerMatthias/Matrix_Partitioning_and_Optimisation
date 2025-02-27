import h5py
import numpy as np
from itertools import combinations
from itertools import chain

def load_h5_file(file, num_cost_matrices):
    """Loads the train and test embeddings"""
    with h5py.File(file, 'r') as f:
        cost_matrices = {i:np.array(f[f'cost_matrix_{i}']) for i in range(num_cost_matrices)}
    return cost_matrices


def creat_sub_matrices_greedy(matrices, vertical_split, horizontal_split):
    if not isinstance(vertical_split, list) or not isinstance(horizontal_split, list): return "Invalid input" # Check if vertical and Horizontal are lists
    vertical_slices = np.vsplit(matrices, vertical_split) #Vertical split
    sub_matrices = list(chain(*[np.hsplit(x, horizontal_split) for x in vertical_slices])) # horizontal Split
    return sub_matrices


len_dataset = 188
num_cost_matrices = len(list(combinations(range(len_dataset), r=2)))

print(f'Number of cost matrices: {num_cost_matrices}')

cost_matrices = load_h5_file('../data/cost_matrices.h5', num_cost_matrices)

print(cost_matrices[0].shape)

all_cost_matrices_sub = []
for i in range(num_cost_matrices):
    all_cost_matrices_sub.append(creat_sub_matrices_greedy(cost_matrices[i], [round(len(cost_matrices[i]) / 2)], [round(len(cost_matrices[i][0]) / 2)]))

print(all_cost_matrices_sub[0][0].shape)
print(all_cost_matrices_sub[0][2].shape)