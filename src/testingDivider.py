import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pandas as pd
from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.MatrixDivider import MatrixDivider

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
MatrixDivider = MatrixDivider()

number_of_matrices = 100
matrix_sizes = [i for i in range(10, 121, 5)]
number_of_runs = 5

lsa_timings = []
lsa_timings_asymmetric = []
matrix_divider_timings = []
matrix_divider_timings_asymmetric = []

lsa_sum = []
lsa_sum_asymmetric = []
divider_sum = []
divider_sum_asymmetric = []

tp_per_run_normal = []
tp_per_run_asymmetric = []

total_start = time.time()
for i in range(number_of_runs):
    start = time.time()
    lsa_timing = []
    lsa_timing_asymmetric = []
    matrix_divider_timing = []
    matrix_divider_timing_asymmetric = []
    for matrix_size in matrix_sizes:
        filepath = f'./data/cost_matrices/cost_matrices_{matrix_size}.h5'
        cost_matrices = MockDataGenerator.load_h5_file(filepath, number_of_matrices)
        lsa_start = time.time()
        lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in
                       range(number_of_matrices)]
        lsa_end = time.time()
        matrix_divider_start = time.time()
        matrix_divider_mapping = MatrixDivider.divider(cost_matrices, number_of_matrices, 4)
        matrix_divider_end = time.time()

        lsa_timing.append(lsa_end - lsa_start)
        matrix_divider_timing.append(matrix_divider_end - matrix_divider_start)

        filepath_2 = f'./data/cost_matrices_asymmetric/cost_matrices_asymmetric_{matrix_size}.h5'
        cost_matrices_2 = MockDataGenerator.load_h5_file(filepath_2, number_of_matrices)
        lsa_asymmetric_start = time.time()
        lsa_asymmetric_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_2[i]) for i in
                                  range(number_of_matrices)]
        lsa_asymmetric_end = time.time()
        matrix_divider_asymmetric_start = time.time()
        matrix_divider_asymmetric_mapping = MatrixDivider.divider(cost_matrices_2, number_of_matrices, 4)
        matrix_divider_asymmetric_end = time.time()
        lsa_timing_asymmetric.append(lsa_asymmetric_end - lsa_asymmetric_start)
        matrix_divider_timing_asymmetric.append(matrix_divider_asymmetric_end - matrix_divider_asymmetric_start)

        # Normal matrices sums
        lsa_cost_sums = []
        divider_cost_sums = []

        for idx in range(number_of_matrices):
            row_lsa, col_lsa = lsa_mapping[idx]
            lsa_cost_sums.append(cost_matrices[idx][row_lsa, col_lsa].sum())

            row_div, col_div = matrix_divider_mapping[idx]
            divider_cost_sums.append(cost_matrices[idx][row_div, col_div].sum())

        lsa_sum.append(lsa_cost_sums)
        divider_sum.append(divider_cost_sums)

        # Asymmetric matrices sums
        lsa_cost_sums_asymmetric = []
        divider_cost_sums_asymmetric = []

        for idx in range(number_of_matrices):
            row_lsa_asym, col_lsa_asym = lsa_asymmetric_mapping[idx]
            lsa_cost_sums_asymmetric.append(cost_matrices_2[idx][row_lsa_asym, col_lsa_asym].sum())

            row_div_asym, col_div_asym = matrix_divider_asymmetric_mapping[idx]
            divider_cost_sums_asymmetric.append(cost_matrices_2[idx][row_div_asym, col_div_asym].sum())

        lsa_sum_asymmetric.append(lsa_cost_sums_asymmetric)
        divider_sum_asymmetric.append(divider_cost_sums_asymmetric)

        # True Positive Rate percentage normal matrices
        tpr_percentages = []
        for idx in range(number_of_matrices):
            lsa_pairs = set(zip(lsa_mapping[idx][0], lsa_mapping[idx][1]))
            div_pairs = set(zip(matrix_divider_mapping[idx][0], matrix_divider_mapping[idx][1]))
            total_assignments = len(lsa_pairs)  # should be equal to matrix size or assignment size
            tp = len(lsa_pairs.intersection(div_pairs))
            tpr_percent = (tp / total_assignments) * 100 if total_assignments > 0 else 0
            tpr_percentages.append(tpr_percent)
        tp_per_run_normal.append(tpr_percentages)

        # True Positive Rate percentage asymmetric matrices
        tpr_percentages_asym = []
        for idx in range(number_of_matrices):
            lsa_pairs_asym = set(zip(lsa_asymmetric_mapping[idx][0], lsa_asymmetric_mapping[idx][1]))
            div_pairs_asym = set(
                zip(matrix_divider_asymmetric_mapping[idx][0], matrix_divider_asymmetric_mapping[idx][1]))
            total_assignments_asym = len(lsa_pairs_asym)
            tp_asym = len(lsa_pairs_asym.intersection(div_pairs_asym))
            tpr_percent_asym = (tp_asym / total_assignments_asym) * 100 if total_assignments_asym > 0 else 0
            tpr_percentages_asym.append(tpr_percent_asym)
        tp_per_run_asymmetric.append(tpr_percentages_asym)

    lsa_timings.append(lsa_timing)
    matrix_divider_timings.append(matrix_divider_timing)
    lsa_timings_asymmetric.append(lsa_timing_asymmetric)
    matrix_divider_timings_asymmetric.append(matrix_divider_timing_asymmetric)

    end = time.time()
    print(f'End of run {i + 1} elapsed time: {end - start}')
total_end = time.time()
print(f'Total elapsed time: {total_end - total_start}')

sns.set(style='whitegrid', palette='deep', font_scale=1.1)

# Convert to numpy arrays
lsa_array = np.array(lsa_timings)
divider_array = np.array(matrix_divider_timings)

# Compute mean and std
lsa_mean = np.mean(lsa_array, axis=0)
lsa_std = np.std(lsa_array, axis=0)

divider_mean = np.mean(divider_array, axis=0)
divider_std = np.std(divider_array, axis=0)

# X axis
x = np.arange(len(matrix_sizes))

# Plot
plt.figure(figsize=(10, 6))

# LSA Plot
plt.plot(x, lsa_mean, label='LSA Mean', color='blue')
plt.fill_between(x, lsa_mean - lsa_std, lsa_mean + lsa_std, color='blue', alpha=0.2)

# Matrix Divider Plot
plt.plot(x, divider_mean, label='Matrix Divider Mean', color='green')
plt.fill_between(x, divider_mean - divider_std, divider_mean + divider_std, color='green', alpha=0.2)


# Axis settings
plt.xticks(x, matrix_sizes, rotation=90)
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title(f'Timing Comparison: LSA vs Matrix Divider for {number_of_runs} runs and {number_of_matrices} matrices')
plt.ylim(0, None)
plt.xlim(0, len(matrix_sizes) - 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filepath = f'./png/{timestamp}.png'
plt.savefig(filepath)
plt.show()

# Convert to numpy arrays
lsa_array_asymmetric = np.array(lsa_timings_asymmetric)
divider_array_asymmetric = np.array(matrix_divider_timings_asymmetric)

# Compute mean and std
lsa_mean_asymmetric = np.mean(lsa_array_asymmetric, axis=0)
lsa_std_asymmetric = np.std(lsa_array_asymmetric, axis=0)

divider_mean_asymmetric = np.mean(divider_array_asymmetric, axis=0)
divider_std_asymmetric = np.std(divider_array_asymmetric, axis=0)

# X axis
x = np.arange(len(matrix_sizes))

# Plot
plt.figure(figsize=(10, 6))

# LSA Plot
plt.plot(x, lsa_mean_asymmetric, label='LSA Mean', color='blue')
plt.fill_between(x, lsa_mean_asymmetric - lsa_std_asymmetric, lsa_mean_asymmetric + lsa_std_asymmetric, color='blue',
                 alpha=0.2)

# Matrix Divider Plot
plt.plot(x, divider_mean_asymmetric, label='Matrix Divider Mean', color='green')
plt.fill_between(x, divider_mean_asymmetric - divider_std_asymmetric, divider_mean_asymmetric + divider_std_asymmetric,
                 color='green', alpha=0.2)


# Axis settings
plt.xticks(x, matrix_sizes, rotation=90)
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title(
    f'Timing Comparison: LSA vs Matrix Divider for {number_of_runs} runs and {number_of_matrices} asymmetric matrices')
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
divider_sum_array = np.array(divider_sum).reshape(number_of_runs, len(matrix_sizes), number_of_matrices)

lsa_sum_asym_array = np.array(lsa_sum_asymmetric).reshape(number_of_runs, len(matrix_sizes), number_of_matrices)
divider_sum_asym_array = np.array(divider_sum_asymmetric).reshape(number_of_runs, len(matrix_sizes), number_of_matrices)

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
plt.title('RMSPE between LSA and Matrix Divider Sums (all datapoints considered)')
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
