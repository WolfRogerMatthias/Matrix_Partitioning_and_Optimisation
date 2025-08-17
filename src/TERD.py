"""
The timing Evaluation for the two real world Datasets

"""

# Imports
import datetime
import time
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tkinter as tk
from pandastable import Table

from itertools import combinations
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.AssignmentAlgo import AssignmentAlgo
from src.BucketAlgo import BucketAlgo
from src.MockDataGenerator import MockDataGenerator
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver

# init

runs = 10
len_mutag = 188
len_ohsu = 79

number_of_matrices_mutag = len(list(combinations(range(len_mutag), r=2)))
number_of_matrices_ohsu = len(list(combinations(range(len_ohsu), r=2)))
number_of_matrices_proteins = 10000

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
AssignmentAlgo = AssignmentAlgo()
MatrixDivider = MatrixDivider()
ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()

cost_matrices_proteins = MockDataGenerator.load_h5_file('./data/cost_matrices_PROTEINS_subset.h5',
                                                        number_of_matrices_proteins)
cost_matrices_mutag = MockDataGenerator.load_h5_file('./data/cost_matrices.h5', number_of_matrices_mutag)
cost_matrices_ohsu = MockDataGenerator.load_h5_file('./data/cost_matrices_OHSU.h5', number_of_matrices_ohsu)

lsa_timing_mutag = []
lsa_timing_ohsu = []
lsa_timing_proteins = []

divider_timing_mutag = []
divider_timing_ohsu = []
divider_timing_proteins = []

bucket_timing_mutag = []
bucket_timing_ohsu = []
bucket_timing_proteins = []

direct_timing_mutag = []
direct_timing_ohsu = []
direct_timing_proteins = []

for i in range(runs):
    start_run = time.time()
    print(f'Start of run {i + 1}, ', end='')

    start_lsa_mutag = time.time()
    lsa_mapping_mutag = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_mutag[j]) for j in
                         range(number_of_matrices_mutag)]
    end_lsa_mutag = time.time()

    start_lsa_ohsu = time.time()
    lsa_mapping_ohsu = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_ohsu[j]) for j in
                        range(number_of_matrices_ohsu)]
    end_lsa_ohsu = time.time()

    start_lsa_proteins = time.time()
    lsa_mapping_proteins = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_proteins[j]) for j in
                            range(number_of_matrices_proteins)]
    end_lsa_proteins = time.time()

    start_divider_mutag = time.time()
    divider_mapping_mutag = MatrixDivider.divider(cost_matrices_mutag, number_of_matrices_mutag, 4)
    end_divider_mutag = time.time()

    start_divider_ohsu = time.time()
    divider_mapping_ohsu = MatrixDivider.divider(cost_matrices_ohsu, number_of_matrices_ohsu, 4)
    end_divider_ohsu = time.time()

    start_divider_proteins = time.time()
    divider_mapping_proteins = MatrixDivider.divider(cost_matrices_proteins, number_of_matrices_proteins, 4)
    end_divider_proteins = time.time()

    start_bucket_mutag = time.time()
    bucket_mapping_mutag = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_mutag, 2)
    end_bucket_mutag = time.time()

    start_bucket_ohsu = time.time()
    bucket_mapping_ohsu = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_ohsu, 2)
    end_bucket_ohsu = time.time()

    start_bucket_proteins = time.time()
    bucket_mapping_proteins = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_proteins, 2)
    end_bucket_proteins = time.time()

    start_direct_mutag = time.time()
    direct_mapping_mutag = [AssignmentAlgo.assignment_applied(cost_matrices_mutag[j]) for j in
                            range(number_of_matrices_mutag)]
    end_direct_mutag = time.time()

    start_direct_ohsu = time.time()
    direct_mapping_ohsu = [AssignmentAlgo.assignment_applied(cost_matrices_ohsu[j]) for j in
                           range(number_of_matrices_ohsu)]
    end_direct_ohsu = time.time()

    start_direct_proteins = time.time()
    direct_mapping_proteins = [AssignmentAlgo.assignment_applied(cost_matrices_proteins[j]) for j in
                               range(number_of_matrices_proteins)]
    end_direct_proteins = time.time()

    lsa_timing_mutag.append(end_lsa_mutag - start_lsa_mutag)
    lsa_timing_ohsu.append(end_lsa_ohsu - start_lsa_ohsu)
    lsa_timing_proteins.append(end_lsa_proteins - start_lsa_proteins)

    divider_timing_mutag.append(end_divider_mutag - start_divider_mutag)
    divider_timing_ohsu.append(end_divider_ohsu - start_divider_ohsu)
    divider_timing_proteins.append(end_divider_proteins - start_divider_proteins)

    bucket_timing_mutag.append(end_bucket_mutag - start_bucket_mutag)
    bucket_timing_ohsu.append(end_bucket_ohsu - start_bucket_ohsu)
    bucket_timing_proteins.append(end_bucket_proteins - start_bucket_proteins)

    direct_timing_mutag.append(end_direct_mutag - start_direct_mutag)
    direct_timing_ohsu.append(end_direct_ohsu - start_direct_ohsu)
    direct_timing_proteins.append(end_direct_proteins - start_direct_proteins)

    end_run_time = time.time()

    print(
        f'end of run time elapsed: {end_run_time - start_run:6.2f}, lsa : {lsa_timing_mutag[-1]:6.2f}, {lsa_timing_ohsu[-1]:6.2f}, divider: {divider_timing_mutag[-1]:6.2f} , {divider_timing_ohsu[-1]:6.2f},  bucket: {bucket_timing_mutag[-1]:6.2f}, {bucket_timing_ohsu[-1]:6.2f}, direct: {direct_timing_mutag[-1]:6.2f}, {direct_timing_ohsu[-1]:6.2f}, proteins: {lsa_timing_proteins[-1]:6.2f}, {divider_timing_proteins[-1]:6.2f}, {bucket_timing_proteins[-1]:6.2f}, {direct_timing_proteins[-1]:6.2f}, {direct_timing_proteins[-1]:6.2f}')

data = {
    "LSA": {
        "Mutag": lsa_timing_mutag,
        "OHSU": lsa_timing_ohsu,
        "Proteins": lsa_timing_proteins
    },
    "Divider": {
        "Mutag": divider_timing_mutag,
        "OHSU": divider_timing_ohsu,
        "Proteins": divider_timing_proteins
    },
    "Bucket": {
        "Mutag": bucket_timing_mutag,
        "OHSU": bucket_timing_ohsu,
        "Proteins": bucket_timing_proteins
    },
    "Direct": {
        "Mutag": direct_timing_mutag,
        "OHSU": direct_timing_ohsu,
        "Proteins": direct_timing_proteins
    }
}

# Build DataFrame
rows = []
for algo, datasets in data.items():
    row = {"Algorithm": algo}
    for ds_name, timings in datasets.items():
        timings_np = np.array(timings)
        row[f"{ds_name}_mean"] = timings_np.mean()
        row[f"{ds_name}_std"] = timings_np.std(ddof=1)  # sample std
        row[f"{ds_name}_min"] = timings_np.min()
        row[f"{ds_name}_max"] = timings_np.max()
    rows.append(row)

df = pd.DataFrame(rows)

# Add speedup & relative runtime compared to LSA
for ds_name in ["Mutag", "OHSU", "Proteins"]:
    lsa_mean = df.loc[df["Algorithm"] == "LSA", f"{ds_name}_mean"].values[0]
    df[f"{ds_name}_speedup_vs_LSA"] = lsa_mean / df[f"{ds_name}_mean"]
    df[f"{ds_name}_runtime_pct_vs_LSA"] = df[f"{ds_name}_mean"] / lsa_mean

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
df.to_csv(f"./png/{timestamp}_timing_results.csv", index=False)
print(df)

# Create main window
root = tk.Tk()
root.title("Timing Results")

# Create a frame to hold the table
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

# Create and show table inside the frame
table = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
table.show()

# Run the Tkinter loop
root.mainloop()
