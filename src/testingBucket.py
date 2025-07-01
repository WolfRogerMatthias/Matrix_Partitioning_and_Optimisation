import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import datetime
import numpy as np

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver
from src.AssignmentAlgo import AssignmentAlgo

if __name__ == "__main__":

    mock_data_generator = MockDataGenerator()
    optimize_algo_applied = OptimizeAlgoApplied()
    parallel_solver = ParallelBucketAssignmentSolver()
    assignment_algo = AssignmentAlgo()

    number_of_matrices = 100
    matrix_sizes = [i for i in range(200, 501, 20)]
    number_of_runs = 5
    bucket_size = 2  # Only bucket size 2 now

    all_lsa_timings = []
    all_bucket_timings = []
    all_tp_rates = []

    all_lsa_costs = []
    all_bucket_costs = []

    all_lsa_timings_asym = []
    all_bucket_timings_asym = []
    all_tp_rates_asym = []

    all_lsa_costs_asym = []
    all_bucket_costs_asym = []

    for run_idx in range(number_of_runs):
        start = time.time()
        print(f"Run {run_idx + 1} / {number_of_runs} ", end='')
        lsa_timing = []
        bucket_timing = []
        tp_rates_run = []
        lsa_costs_run = []
        bucket_costs_run = []

        lsa_timing_asym = []
        bucket_timing_asym = []
        tp_rates_run_asym = []
        lsa_costs_run_asym = []
        bucket_costs_run_asym = []

        for matrix_size in matrix_sizes:
            cost_matrices = mock_data_generator.load_h5_file(
                f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                number_of_matrices
            )
            cost_matrix_dict = {f"matrix_{i}": cost_matrices[i] for i in range(number_of_matrices)}

            # LSA baseline timing
            lsa_start = time.time()
            lsa_mapping = [
                optimize_algo_applied.compute_linear_sum_assignment(cost_matrices[i])
                for i in range(number_of_matrices)
            ]
            lsa_end = time.time()
            lsa_timing.append(lsa_end - lsa_start)

            # Parallel bucket solver timing (bucket size = 2)
            bucket_start = time.time()
            bucket_mapping = parallel_solver.solve_multiple(cost_matrix_dict, bucket_size)
            bucket_end = time.time()
            bucket_timing.append(bucket_end - bucket_start)

            # Calculate TP rate per matrix
            lsa_pairs_list = [set(zip(r, c)) for r, c in lsa_mapping]
            bucket_pairs_list = [set(bucket_mapping[f"matrix_{i}"]) for i in range(number_of_matrices)]

            tp_counts = [len(lsa_pairs_list[i] & bucket_pairs_list[i]) for i in range(number_of_matrices)]
            tp_rates = [tp_counts[i] / len(lsa_pairs_list[i]) for i in range(number_of_matrices)]
            tp_rates_run.append(tp_rates)

            # Calculate minimal sums for each matrix
            lsa_costs = []
            bucket_costs = []
            for i in range(number_of_matrices):
                lsa_rows, lsa_cols = lsa_mapping[i]
                bucket_pairs = bucket_mapping[f"matrix_{i}"]

                # LSA minimal sum
                lsa_sum = cost_matrices[i][lsa_rows, lsa_cols].sum()
                lsa_costs.append(lsa_sum)

                # Parallel Bucket minimal sum (sum costs of assigned pairs)
                bucket_sum = sum(cost_matrices[i][r, c] for r, c in bucket_pairs)
                bucket_costs.append(bucket_sum)

            lsa_costs_run.append(lsa_costs)
            bucket_costs_run.append(bucket_costs)

            # Load asymmetric cost matrices
            cost_matrices_asym = mock_data_generator.load_h5_file(
                f'./data/cost_matrices_asymmetric/cost_matrices_asymmetric_{matrix_size}.h5',
                number_of_matrices
            )
            cost_matrix_dict_asym = {f"matrix_{i}": cost_matrices_asym[i] for i in range(number_of_matrices)}

            # LSA baseline timing for asymmetric matrices
            lsa_start_asym = time.time()
            lsa_mapping_asym = [
                optimize_algo_applied.compute_linear_sum_assignment(cost_matrices_asym[i])
                for i in range(number_of_matrices)
            ]
            lsa_end_asym = time.time()
            lsa_timing_asym.append(lsa_end_asym - lsa_start_asym)

            # Parallel bucket solver timing for asymmetric matrices
            bucket_start_asym = time.time()
            bucket_mapping_asym = parallel_solver.solve_multiple(cost_matrix_dict_asym, bucket_size)
            bucket_end_asym = time.time()
            bucket_timing_asym.append(bucket_end_asym - bucket_start_asym)

            # Calculate TP rate per asymmetric matrix
            lsa_pairs_list_asym = [set(zip(r, c)) for r, c in lsa_mapping_asym]
            bucket_pairs_list_asym = [set(bucket_mapping_asym[f"matrix_{i}"]) for i in range(number_of_matrices)]

            tp_counts_asym = [len(lsa_pairs_list_asym[i] & bucket_pairs_list_asym[i]) for i in
                              range(number_of_matrices)]
            tp_rates_asym = [tp_counts_asym[i] / len(lsa_pairs_list_asym[i]) for i in range(number_of_matrices)]
            tp_rates_run_asym.append(tp_rates_asym)

            # Calculate minimal sums for each asymmetric matrix
            lsa_costs_asym = []
            bucket_costs_asym = []
            for i in range(number_of_matrices):
                lsa_rows_asym, lsa_cols_asym = lsa_mapping_asym[i]
                bucket_pairs_asym = bucket_mapping_asym[f"matrix_{i}"]

                lsa_sum_asym = cost_matrices_asym[i][lsa_rows_asym, lsa_cols_asym].sum()
                lsa_costs_asym.append(lsa_sum_asym)

                bucket_sum_asym = sum(cost_matrices_asym[i][r, c] for r, c in bucket_pairs_asym)
                bucket_costs_asym.append(bucket_sum_asym)

            lsa_costs_run_asym.append(lsa_costs_asym)
            bucket_costs_run_asym.append(bucket_costs_asym)

        all_lsa_timings.append(lsa_timing)
        all_bucket_timings.append(bucket_timing)
        all_tp_rates.append(tp_rates_run)
        all_lsa_costs.append(lsa_costs_run)
        all_bucket_costs.append(bucket_costs_run)

        all_lsa_timings_asym.append(lsa_timing_asym)
        all_bucket_timings_asym.append(bucket_timing_asym)
        all_tp_rates_asym.append(tp_rates_run_asym)
        all_lsa_costs_asym.append(lsa_costs_run_asym)
        all_bucket_costs_asym.append(bucket_costs_run_asym)

        end = time.time()
        print(f"Runtime: {end - start:6.2f} seconds")
    # Convert to numpy arrays for stats
    lsa_array = np.array(all_lsa_timings)
    bucket_array = np.array(all_bucket_timings)
    tp_array = np.array(all_tp_rates)  # shape: (runs, matrix_sizes, number_of_matrices)
    lsa_costs_array = np.array(all_lsa_costs)  # shape: (runs, matrix_sizes, number_of_matrices)
    bucket_costs_array = np.array(all_bucket_costs)

    x = np.arange(len(matrix_sizes))

    # --- Timing plot ---
    plt.figure(figsize=(12, 7))
    plt.plot(x, np.mean(lsa_array, axis=0), label='LSA Mean', color='blue')
    plt.fill_between(
        x,
        np.mean(lsa_array, axis=0) - np.std(lsa_array, axis=0),
        np.mean(lsa_array, axis=0) + np.std(lsa_array, axis=0),
        color='blue',
        alpha=0.2
    )

    plt.plot(x, np.mean(bucket_array, axis=0), label='Parallel Bucket Mean (buckets=2)', color='green')
    plt.fill_between(
        x,
        np.mean(bucket_array, axis=0) - np.std(bucket_array, axis=0),
        np.mean(bucket_array, axis=0) + np.std(bucket_array, axis=0),
        color='green',
        alpha=0.2
    )

    plt.xticks(x, matrix_sizes, rotation=90)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title(
        f'Timing Comparison: LSA vs Parallel Bucket (Bucket Size=2)\n{number_of_runs} runs, {number_of_matrices} matrices per size')
    plt.ylim(0, None)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')
    plt.show()
    """
    lsa_array_asym = np.array(all_lsa_timings_asym)
    bucket_array_asym = np.array(all_bucket_timings_asym)
    plt.figure(figsize=(12, 7))
    plt.plot(x, np.mean(lsa_array_asym, axis=0), label='LSA Mean Asymmetric', color='purple')
    plt.fill_between(
        x,
        np.mean(lsa_array_asym, axis=0) - np.std(lsa_array_asym, axis=0),
        np.mean(lsa_array_asym, axis=0) + np.std(lsa_array_asym, axis=0),
        color='purple',
        alpha=0.2
    )

    plt.plot(x, np.mean(bucket_array_asym, axis=0), label='Parallel Bucket Mean Asymmetric (buckets=2)', color='orange')
    plt.fill_between(
        x,
        np.mean(bucket_array_asym, axis=0) - np.std(bucket_array_asym, axis=0),
        np.mean(bucket_array_asym, axis=0) + np.std(bucket_array_asym, axis=0),
        color='orange',
        alpha=0.2
    )

    plt.xticks(x, matrix_sizes, rotation=90)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title(
        f'Timing Comparison Asymmetric: LSA vs Parallel Bucket (Bucket Size=2)\n{number_of_runs} runs, {number_of_matrices} matrices per size')
    plt.ylim(0, None)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')
    plt.show()

    # --- Prepare TP rate dataframe for boxplot ---
    tp_data = []
    for run_idx in range(tp_array.shape[0]):
        for size_idx in range(tp_array.shape[1]):
            for matrix_idx in range(tp_array.shape[2]):
                tp_data.append({
                    "Matrix Size": matrix_sizes[size_idx],
                    "TP Rate": tp_array[run_idx][size_idx][matrix_idx],
                    "Run": run_idx,
                    "Matrix": matrix_idx,
                    "Type": "Normal"
                })
    tp_array_asym = np.array(all_tp_rates_asym)

    tp_data_asym = []
    for run_idx in range(tp_array_asym.shape[0]):
        for size_idx in range(tp_array_asym.shape[1]):
            for matrix_idx in range(tp_array_asym.shape[2]):
                tp_data_asym.append({
                    "Matrix Size": matrix_sizes[size_idx],
                    "TP Rate": tp_array_asym[run_idx][size_idx][matrix_idx],
                    "Run": run_idx,
                    "Matrix": matrix_idx,
                    "Type": "Asymmetric"
                })

    tp_data_df = pd.DataFrame(tp_data)
    tp_data_asym_df = pd.DataFrame(tp_data_asym)
    tp_df = pd.concat([tp_data_df, tp_data_asym_df])

    # --- Seaborn boxplot for TP rate ---
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="Matrix Size", y="TP Rate", hue="Type", data=tp_df,
                palette={"Normal": "violet", "Asymmetric": "orange"})
    plt.title('True Positive Rate of Parallel Bucket Algorithm (Bucket Size=2) by Matrix Size')
    plt.ylabel('TP Rate')
    plt.xlabel('Matrix Size')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')
    plt.show()



    # Assuming lsa_costs_array, bucket_costs_array, lsa_costs_array_asym, bucket_costs_array_asym, matrix_sizes already defined

    # --- Calculate RMSPE ---
    def calculate_rmspe(lsa_costs, bucket_costs):
        error_data = []
        for run_idx in range(lsa_costs.shape[0]):
            for size_idx in range(lsa_costs.shape[1]):
                for matrix_idx in range(lsa_costs.shape[2]):
                    true_cost = lsa_costs[run_idx][size_idx][matrix_idx]
                    pred_cost = bucket_costs[run_idx][size_idx][matrix_idx]
                    # Avoid division by zero
                    if true_cost != 0:
                        perc_error_sq = ((pred_cost - true_cost) / true_cost) ** 2
                        error_data.append({
                            "Matrix Size": matrix_sizes[size_idx],
                            "Perc Error Squared": perc_error_sq
                        })
        error_df = pd.DataFrame(error_data)
        grouped = error_df.groupby("Matrix Size")
        rmspe = np.sqrt(grouped["Perc Error Squared"].mean()) * 100  # Multiply by 100 for percentage
        return rmspe


    lsa_costs_array_asym = np.array(all_lsa_costs_asym)
    bucket_costs_array_asym = np.array(all_bucket_costs_asym)

    rmspe_normal = calculate_rmspe(lsa_costs_array, bucket_costs_array)
    rmspe_asym = calculate_rmspe(lsa_costs_array_asym, bucket_costs_array_asym)

    # --- Plotting RMSPE ---
    plt.figure(figsize=(12, 6))
    plt.plot(matrix_sizes, rmspe_normal.values, marker='o', color='blue', label='RMSPE Normal Matrices')
    plt.plot(matrix_sizes, rmspe_asym.values, marker='s', color='orange', label='RMSPE Asymmetric Matrices')
    plt.xlabel("Matrix Size")
    plt.ylabel("Root Mean Squared Percentage Error (%)")
    plt.title("RMSPE Between LSA and Parallel Bucket Solver by Matrix Size")
    plt.xticks(matrix_sizes, rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')
    plt.show()

    selected_sizes = [200, 300, 400]

    for size in selected_sizes:
        plt.figure(figsize=(8, 6))
        size_idx = matrix_sizes.index(size)

        # Flatten runs and matrices minimal sums for this size
        lsa_vals = lsa_costs_array[:, size_idx, :].flatten()
        bucket_vals = bucket_costs_array[:, size_idx, :].flatten()

        plt.scatter(lsa_vals, bucket_vals, alpha=0.6, color='purple')

        # Compute overall min and max for axes
        overall_min = min(lsa_vals.min(), bucket_vals.min())
        overall_max = max(lsa_vals.max(), bucket_vals.max())

        # Set axis limits with a small margin
        margin = (overall_max - overall_min) * 0.05
        plt.xlim(overall_min - margin, overall_max + margin)
        plt.ylim(overall_min - margin, overall_max + margin)

        # Plot y=x line for reference
        plt.plot([overall_min - margin, overall_max + margin], [overall_min - margin, overall_max + margin],
                 'k--', label='y = x (Ideal)')

        plt.xlabel("LSA Minimal Sum")
        plt.ylabel("Parallel Bucket Minimal Sum")
        plt.title(f"Scatter Plot of Minimal Sums for Matrix Size {size}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        plt.savefig(f'./png/{timestamp}_{size}.png')
        plt.show()

    for size in selected_sizes:
        plt.figure(figsize=(8, 6))
        size_idx = matrix_sizes.index(size)

        lsa_vals_asym = lsa_costs_array_asym[:, size_idx, :].flatten()
        bucket_vals_asym = bucket_costs_array_asym[:, size_idx, :].flatten()

        plt.scatter(lsa_vals_asym, bucket_vals_asym, alpha=0.6, color='darkorange')

        overall_min = min(lsa_vals_asym.min(), bucket_vals_asym.min())
        overall_max = max(lsa_vals_asym.max(), bucket_vals_asym.max())
        margin = (overall_max - overall_min) * 0.05
        plt.xlim(overall_min - margin, overall_max + margin)
        plt.ylim(overall_min - margin, overall_max + margin)

        plt.plot([overall_min - margin, overall_max + margin], [overall_min - margin, overall_max + margin],
                 'k--', label='y = x (Ideal)')

        plt.xlabel("LSA Minimal Sum (Asymmetric)")
        plt.ylabel("Parallel Bucket Minimal Sum (Asymmetric)")
        plt.title(f"Scatter Plot of Minimal Sums for Matrix Size {size} (Asymmetric)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        plt.savefig(f'./png/{timestamp}_{size}.png')
        plt.show()
"""