import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pandas as pd
from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.AssignmentAlgo import AssignmentAlgo

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
AssignmentAlgo = AssignmentAlgo()

number_of_matrices = 100
matrix_sizes = [i for i in range(150, 251, 5)]
number_of_runs = 5

lsa_timings = []
lsa_timings_asymmetric = []
assignment_timings = []
assignment_timings_asymmetric = []

lsa_sum = []
lsa_sum_asymmetric = []
assignment_sum = []
assignment_sum_asymmetric = []

tp_per_run_normal = []
tp_per_run_asymmetric = []

for i in range(number_of_runs):
    start = time.time()
    lsa_timing = []
    lsa_timing_asymmetric = []
    assignment_timing = []
    assignment_timing_asymmetric = []
    for matrix_size in matrix_sizes:
        filepath = f'./data/cost_matrices/cost_matrices_{matrix_size}.h5'
        cost_matrices = MockDataGenerator.load_h5_file(filepath, number_of_matrices)
        lsa_start = time.time()
        lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in
                       range(number_of_matrices)]
        lsa_end = time.time()
        lsa_timing.append(lsa_end - lsa_start)

        assignment_start = time.time()
        assignment_mapping = [AssignmentAlgo.assignment_applied(cost_matrices[i]) for i in range(number_of_matrices)]
        assignment_end = time.time()
        assignment_timing.append(assignment_end - assignment_start)

        filepath_asym = f'./data/cost_matrices_asymmetric/cost_matrices_asymmetric_{matrix_size}.h5'
        cost_matrices_asymmetric = MockDataGenerator.load_h5_file(filepath_asym, number_of_matrices)

        lsa_start_asym = time.time()
        lsa_mapping_asym = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_asymmetric[i]) for i in
                            range(number_of_matrices)]
        lsa_end_asym = time.time()
        lsa_timing_asymmetric.append(lsa_end_asym - lsa_start)

        assignment_start_asym = time.time()
        assignment_mapping_asym = [AssignmentAlgo.assignment_applied(cost_matrices_asymmetric[i]) for i in
                                   range(number_of_matrices)]
        assignment_end_asym = time.time()
        assignment_timing_asymmetric.append(assignment_end_asym - assignment_start)

        lsa_cost_sums = []
        assign_cost_sums = []
        lsa_cost_asymmetric = []
        assign_cost_asymmetric = []

        tpr_percentages = []
        tpr_percentages_asymmetric = []

        for idx in range(number_of_matrices):
            row_lsa, col_lsa = lsa_mapping[idx]
            lsa_cost_sums.append(cost_matrices[idx][row_lsa, col_lsa].sum())

            assignment = assignment_mapping[idx]
            assign_cost_sums.append(sum(cost_matrices[idx][row, col] for row, col in assignment.items()))

            row_lsa_asym, col_lsa_asym = lsa_mapping_asym[idx]
            lsa_cost_asymmetric.append(cost_matrices_asymmetric[idx][row_lsa_asym, col_lsa_asym].sum())

            assignment_asym = assignment_mapping_asym[idx]
            assign_cost_asymmetric.append(
                sum(cost_matrices_asymmetric[idx][row, col] for row, col in assignment_asym.items()))

            lsa_pairs = set(zip(lsa_mapping[idx][0], lsa_mapping[idx][1]))
            assign_pairs = set(assignment_mapping[idx].items())
            tpr_perc = len(lsa_pairs.intersection(assign_pairs)) / len(lsa_pairs) * 100
            tpr_percentages.append(tpr_perc)

            lsa_pairs_asym = set(zip(lsa_mapping_asym[idx][0], lsa_mapping_asym[idx][1]))
            assign_pairs_asym = set(assignment_mapping_asym[idx].items())
            tpr_perc_asym = len(lsa_pairs_asym.intersection(assign_pairs_asym)) / len(lsa_pairs_asym) * 100
            tpr_percentages_asymmetric.append(tpr_perc_asym)

        lsa_sum.append(lsa_cost_sums)
        lsa_sum_asymmetric.append(lsa_cost_asymmetric)
        assignment_sum.append(assign_cost_sums)
        assignment_sum_asymmetric.append(assign_cost_asymmetric)
        tp_per_run_normal.append(tpr_percentages)
        tp_per_run_asymmetric.append(tpr_percentages_asymmetric)

    lsa_timings.append(lsa_timing)
    assignment_timings.append(assignment_timing)
    lsa_timings_asymmetric.append(lsa_timing_asymmetric)
    assignment_timings_asymmetric.append(assignment_timing_asymmetric)
    end = time.time()
    print(f'End of run {i + 1}, time elapsed: {end - start:6.2f}')

sns.set(style='whitegrid', palette='deep', font_scale=1.1)

lsa_array = np.array(lsa_timings)
assignment_array = np.array(assignment_timings)

lsa_mean = np.mean(lsa_array, axis=0)
lsa_std = np.std(lsa_array, axis=0)

assignment_mean = np.mean(assignment_array, axis=0)
assignment_std = np.std(assignment_array, axis=0)

x = np.arange(len(matrix_sizes))

plt.figure(figsize=[10, 6])

plt.plot(x, lsa_mean, label='LSA Mean', color='blue')
plt.fill_between(x, lsa_mean - lsa_std, lsa_mean + lsa_std, alpha=0.2, color='blue')

plt.plot(x, assignment_mean, label='Assignment Mean', color='green')
plt.fill_between(x, assignment_mean - assignment_std, assignment_mean + assignment_std, alpha=0.2, color='green')

# Axis settings
plt.xticks(x, matrix_sizes, rotation=90)
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title(f'Timing Comparison: LSA vs Assignment for {number_of_runs} runs and {number_of_matrices} matrices')
plt.ylim(0, None)
plt.xlim(0, len(matrix_sizes) - 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
time.sleep(1)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filepath = f'./png/{timestamp}.png'
plt.savefig(filepath)
plt.show()

lsa_array_asymmetric = np.array(lsa_timings_asymmetric)
assignment_array_asymmetric = np.array(assignment_timings_asymmetric)

lsa_mean_asymmetric = np.mean(lsa_array_asymmetric, axis=0)
lsa_std_asymmetric = np.std(lsa_array_asymmetric, axis=0)

assignment_mean_asymmetric = np.mean(assignment_array_asymmetric, axis=0)
assignment_std_asymmetric = np.std(assignment_array_asymmetric, axis=0)

plt.figure(figsize=[10, 6])
plt.plot(x, lsa_mean_asymmetric, label='LSA Mean', color='blue')
plt.fill_between(x, lsa_mean_asymmetric - lsa_std_asymmetric, lsa_mean_asymmetric + lsa_std_asymmetric, alpha=0.2,
                 color='blue')

plt.plot(x, assignment_mean_asymmetric, label='Assignment Mean', color='green')
plt.fill_between(x, assignment_mean_asymmetric - assignment_std_asymmetric,
                 assignment_mean_asymmetric + assignment_std_asymmetric, alpha=0.2, color='green')

# Axis settings
plt.xticks(x, matrix_sizes, rotation=90)
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title(
    f'Timing Comparison: LSA vs Assignment for {number_of_runs} runs and {number_of_matrices} asymmetric matrices')
plt.ylim(0, None)
plt.xlim(0, len(matrix_sizes) - 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
time.sleep(1)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filepath = f'./png/{timestamp}.png'
plt.savefig(filepath)
plt.show()

# Convert the raw sums lists to numpy arrays and reshape to (runs, matrix_sizes, number_of_matrices)
lsa_sum_array = np.array(lsa_sum).reshape(number_of_runs, len(matrix_sizes), number_of_matrices)
divider_sum_array = np.array(assignment_sum).reshape(number_of_runs, len(matrix_sizes), number_of_matrices)

lsa_sum_asym_array = np.array(lsa_sum_asymmetric).reshape(number_of_runs, len(matrix_sizes), number_of_matrices)
divider_sum_asym_array = np.array(assignment_sum_asymmetric).reshape(number_of_runs, len(matrix_sizes),
                                                                     number_of_matrices)

# Flatten runs and matrices axes into one combined axis per matrix size:
# New shape: (number_of_runs * number_of_matrices, matrix_sizes)
lsa_vals_per_size = lsa_sum_array.reshape(number_of_runs * number_of_matrices, len(matrix_sizes))
divider_vals_per_size = divider_sum_array.reshape(number_of_runs * number_of_matrices, len(matrix_sizes))

lsa_vals_per_size_asym = lsa_sum_asym_array.reshape(number_of_runs * number_of_matrices, len(matrix_sizes))
divider_vals_per_size_asym = divider_sum_asym_array.reshape(number_of_runs * number_of_matrices, len(matrix_sizes))

# Compute RMSE per matrix size across all runs and matrices combined
rmse = np.sqrt(np.mean((lsa_vals_per_size - divider_vals_per_size) ** 2, axis=0))
rmse_asym = np.sqrt(np.mean((lsa_vals_per_size_asym - divider_vals_per_size_asym) ** 2, axis=0))

# Compute mean LSA sums per matrix size for normalization
mean_lsa = np.mean(lsa_vals_per_size, axis=0)
mean_lsa_asym = np.mean(lsa_vals_per_size_asym, axis=0)

# Compute pRMSE (%)
pRMSE = (rmse / mean_lsa) * 100
pRMSE_asym = (rmse_asym / mean_lsa_asym) * 100

# Plot the pRMSE curves
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, pRMSE, label='Normal Matrices pRMSE', color='red')
plt.plot(matrix_sizes, pRMSE_asym, label='Asymmetric Matrices pRMSE', color='orange')

plt.xlabel('Matrix Size')
plt.ylabel('RMSPE (%)')
plt.title('RMSPE between LSA and Assignment Sums (all datapoints considered)')
plt.legend()
plt.grid(True)
plt.ylim(0, None)
plt.tight_layout()

# Save figure with timestamp
time.sleep(1)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filepath = f'./png/{timestamp}.png'
plt.savefig(filepath)
plt.show()

num_runs = number_of_runs  # your number_of_runs
num_sizes = len(matrix_sizes)  # number of matrix sizes

# Initialize dictionary to collect data
data = {
    'Matrix Size': [],
    'TPR (%)': [],
    'Type': []
}

# Process normal matrices
for run in range(num_runs):
    for size_idx in range(num_sizes):
        # Calculate the index in the flattened tp list
        idx = run * num_sizes + size_idx
        matrix_size = matrix_sizes[size_idx]

        # Append all values for this matrix size and run
        for val in tp_per_run_normal[idx]:
            data['Matrix Size'].append(matrix_size)
            data['TPR (%)'].append(val)
            data['Type'].append('Normal')

# Process asymmetric matrices similarly
for run in range(num_runs):
    for size_idx in range(num_sizes):
        idx = run * num_sizes + size_idx
        matrix_size = matrix_sizes[size_idx]

        for val in tp_per_run_asymmetric[idx]:
            data['Matrix Size'].append(matrix_size)
            data['TPR (%)'].append(val)
            data['Type'].append('Asymmetric')

# Create DataFrame
df = pd.DataFrame(data)

# Plotting boxplot side-by-side
plt.figure(figsize=(14, 7))
sns.boxplot(x='Matrix Size', y='TPR (%)', hue='Type', data=df, palette=['#1f77b4', '#ff7f0e'])

plt.title('True Positive Rate Percentage (TPR%) per Matrix Size (All Runs Combined)')
plt.xticks(rotation=90)
plt.xlabel('Matrix Size')
plt.ylabel('TPR (%)')
plt.legend(title='Matrix Type')
plt.ylim(-0.25, 100.25)
plt.tight_layout()
time.sleep(1)
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
plt.savefig(f'./png/{timestamp}.png')
plt.show()
