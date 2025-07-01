from itertools import combinations, chain
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import numpy as np
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.AssignmentAlgo import AssignmentAlgo
from src.BucketAlgo import BucketAlgo
from src.MatrixDivider import MatrixDivider
from src.MockDataGenerator import MockDataGenerator
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver

if __name__ == '__main__':
    runs = 5
    len_mutag = 188
    len_ohsu = 79

    number_of_matrices_mutag = len(list(combinations(range(len_mutag), r=2)))
    number_of_matrices_ohsu = len(list(combinations(range(len_ohsu), r=2)))

    MockDataGenerator = MockDataGenerator()
    OptimizeAlgoApplied = OptimizeAlgoApplied()
    AssignmentAlgo = AssignmentAlgo()
    BucketAlgo = BucketAlgo(OptimizeAlgoApplied)
    MatrixDivider = MatrixDivider()
    ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()

    cost_matrices_mutag = MockDataGenerator.load_h5_file('./data/cost_matrices.h5', number_of_matrices_mutag)
    cost_matrices_ohsu = MockDataGenerator.load_h5_file('./data/cost_matrices_OHSU.h5', number_of_matrices_ohsu)

    lsap_timings_mutag = []
    assignment_timings_mutag = []
    divider_timings_mutag = []
    parallel_timings_mutag = []

    lsap_minimal_sum_mutag = []
    assignment_minimal_sum_mutag = []
    divider_minimal_sum_mutag = []
    parallel_minimal_sum_mutag = []

    assign_true_positives_mutag = []
    divider_true_positives_mutag = []
    parallel_true_positives_mutag = []

    lsap_timings_ohsu = []
    assignment_timings_ohsu = []
    divider_timings_ohsu = []
    parallel_timings_ohsu = []

    lsap_minimal_sum_ohsu = []
    assignment_minimal_sum_ohsu = []
    divider_minimal_sum_ohsu = []
    parallel_minimal_sum_ohsu = []

    assign_true_positives_ohsu = []
    divider_true_positives_ohsu = []
    parallel_true_positives_ohsu = []

    start_of_runs = time.time()
    print(f'Start of {runs} runs')
    for i in range(runs):
        start = time.time()

        start_lsap_mutag = time.time()
        lsap_mapping_mutag = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_mutag[i]) for i in
                              range(number_of_matrices_mutag)]
        end_lsap_mutag = time.time()
        start_assignment_mutag = time.time()
        assignment_mapping_mutag = [AssignmentAlgo.assignment_applied(cost_matrices_mutag[i]) for i in
                                    range(number_of_matrices_mutag)]
        end_assignment_mutag = time.time()
        start_divider_mutag = time.time()
        divider_mapping_mutag = MatrixDivider.divider(cost_matrices_mutag, number_of_matrices_mutag, 4)
        end_divider_mutag = time.time()

        start_parallel_mutag = time.time()
        parallel_mapping_mutag = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_mutag, 2)
        end_parallel_mutag = time.time()

        lsap_timings_mutag.append(end_lsap_mutag - start_lsap_mutag)
        assignment_timings_mutag.append(end_assignment_mutag - start_assignment_mutag)
        divider_timings_mutag.append(end_divider_mutag - start_divider_mutag)
        parallel_timings_mutag.append(end_parallel_mutag - start_parallel_mutag)

        start_lsap_ohsu = time.time()
        lsap_mapping_ohsu = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices_ohsu[i]) for i in
                             range(number_of_matrices_ohsu)]
        end_lsap_ohsu = time.time()
        start_assignment_ohsu = time.time()
        assignment_mapping_ohsu = [AssignmentAlgo.assignment_applied(cost_matrices_ohsu[i]) for i in
                                   range(number_of_matrices_ohsu)]
        end_assignment_ohsu = time.time()
        start_divider_ohsu = time.time()
        divider_mapping_ohsu = MatrixDivider.divider(cost_matrices_ohsu, number_of_matrices_ohsu, 4)
        end_divider_ohsu = time.time()
        start_parallel_ohsu = time.time()
        parallel_mapping_ohsu = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices_ohsu, 2)
        end_parallel_ohsu = time.time()

        lsap_timings_ohsu.append(end_lsap_ohsu - start_lsap_ohsu)
        assignment_timings_ohsu.append(end_assignment_ohsu - start_assignment_ohsu)
        divider_timings_ohsu.append(end_divider_ohsu - start_divider_ohsu)
        parallel_timings_ohsu.append(end_parallel_ohsu - start_parallel_ohsu)

        # Calculation of minimal sum for datasets
        mutag_sum_lsap = []
        mutag_sum_assign = []
        mutag_sum_divider = []
        mutag_sum_parallel = []

        for idx, key in enumerate(cost_matrices_mutag.keys()):
            matrix = cost_matrices_mutag[key]

            # Lsap Sum
            row_ind, col_ind = lsap_mapping_mutag[idx]
            mutag_sum_lsap.append(matrix[row_ind, col_ind].sum())

            # Assignment Sum
            assignment = assignment_mapping_mutag[idx]
            mutag_sum_assign.append(sum(matrix[row, col] for row, col in assignment.items()))

            # Divider Sum
            divider_row_ind = divider_mapping_mutag[idx][0]
            divider_col_ind = divider_mapping_mutag[idx][1]
            mutag_sum_divider.append(matrix[divider_row_ind, divider_col_ind].sum())

            bucket_pairs = parallel_mapping_mutag[idx]
            bucket_sum = sum(matrix[r, c] for r, c in bucket_pairs)
            mutag_sum_parallel.append(bucket_sum)

        lsap_minimal_sum_mutag.append(mutag_sum_lsap)
        assignment_minimal_sum_mutag.append(mutag_sum_assign)
        divider_minimal_sum_mutag.append(mutag_sum_divider)
        parallel_minimal_sum_mutag.append(mutag_sum_parallel)

        ohsu_sum_lsap = []
        ohsu_sum_assign = []
        ohsu_sum_divider = []
        ohsu_sum_parallel = []

        for idx, key in enumerate(cost_matrices_ohsu.keys()):
            matrix = cost_matrices_ohsu[key]

            # Lsap Sum
            row_ind, col_ind = lsap_mapping_ohsu[idx]
            ohsu_sum_lsap.append(matrix[row_ind, col_ind].sum())

            # Assignment Sum
            assignment = assignment_mapping_ohsu[idx]
            ohsu_sum_assign.append(sum(matrix[row, col] for row, col in assignment.items()))

            # Divider Sum
            divider_row_ind = divider_mapping_ohsu[idx][0]
            divider_col_ind = divider_mapping_ohsu[idx][1]
            ohsu_sum_divider.append(matrix[divider_row_ind, divider_col_ind].sum())

            bucket_pairs = parallel_mapping_ohsu[idx]
            bucket_sum = sum(matrix[r, c] for r, c in bucket_pairs)
            ohsu_sum_parallel.append(bucket_sum)

        lsap_minimal_sum_ohsu.append(ohsu_sum_lsap)
        assignment_minimal_sum_ohsu.append(ohsu_sum_assign)
        divider_minimal_sum_ohsu.append(ohsu_sum_divider)
        parallel_minimal_sum_ohsu.append(ohsu_sum_parallel)

        # Calculation of true positivs for datasets
        assign_tp_per = []
        divider_tp_per = []

        for baseline, assign, divider in zip(lsap_mapping_mutag, assignment_mapping_mutag, divider_mapping_mutag):
            baseline_pairs = set(zip(*baseline))
            assign_pairs = set(assign.items())
            divider_pairs = set(zip(*divider))

            total = len(baseline_pairs)
            if total == 0:
                # Avoid division by zero
                assign_tp_per.append(0.0)
                divider_tp_per.append(0.0)
                continue

            assign_tp = len(baseline_pairs.intersection(assign_pairs))
            divider_tp = len(baseline_pairs.intersection(divider_pairs))

            assign_tp_per.append((assign_tp / total) * 100)
            divider_tp_per.append((divider_tp / total) * 100)

        assign_true_positives_mutag.append(assign_tp_per)
        divider_true_positives_mutag.append(divider_tp_per)

        parallel_tp_per = []
        for idx, baseline in enumerate(lsap_mapping_mutag):
            baseline_pairs = set(zip(*baseline))
            parallel_pairs = set(parallel_mapping_mutag[idx])  # parallel results are pairs like (row, col)

            total = len(baseline_pairs)
            if total == 0:
                parallel_tp_per.append(0.0)
                continue
            parallel_tp = len(baseline_pairs.intersection(parallel_pairs))
            parallel_tp_per.append((parallel_tp / total) * 100)

        parallel_true_positives_mutag.append(parallel_tp_per)

        assign_tp_per_ohsu = []
        divider_tp_per_ohsu = []

        for baseline, assign, divider in zip(lsap_mapping_ohsu, assignment_mapping_ohsu, divider_mapping_ohsu):
            baseline_pairs = set(zip(*baseline))
            assign_pairs = set(assign.items())
            divider_pairs = set(zip(*divider))

            total = len(baseline_pairs)
            if total == 0:
                # Avoid division by zero
                assign_tp_per_ohsu.append(0.0)
                divider_tp_per_ohsu.append(0.0)
                continue

            assign_tp = len(baseline_pairs.intersection(assign_pairs))
            divider_tp = len(baseline_pairs.intersection(divider_pairs))

            assign_tp_per_ohsu.append((assign_tp / total) * 100)
            divider_tp_per_ohsu.append((divider_tp / total) * 100)

        assign_true_positives_ohsu.append(assign_tp_per_ohsu)
        divider_true_positives_ohsu.append(divider_tp_per_ohsu)

        parallel_tp_per_ohsu = []
        for idx, baseline in enumerate(lsap_mapping_ohsu):
            baseline_pairs = set(zip(*baseline))
            parallel_pairs = set(parallel_mapping_ohsu[idx])

            total = len(baseline_pairs)
            if total == 0:
                parallel_tp_per_ohsu.append(0.0)
                continue
            parallel_tp = len(baseline_pairs.intersection(parallel_pairs))
            parallel_tp_per_ohsu.append((parallel_tp / total) * 100)

        parallel_true_positives_ohsu.append(parallel_tp_per_ohsu)

        end = time.time()
        print(f'Run {i + 1} finished in {end - start} seconds')
    end_of_runs = time.time()
    print(f'End of {runs} runs, time elapsed: {end_of_runs - start_of_runs:6.2f}')

    # Prepare the DataFrame
    data = {
        'Time': lsap_timings_mutag + assignment_timings_mutag + divider_timings_mutag + parallel_timings_mutag +
                lsap_timings_ohsu + assignment_timings_ohsu + divider_timings_ohsu + parallel_timings_ohsu,

        'Algorithm': (['LSA'] * len(lsap_timings_mutag) +
                      ['Divider'] * len(divider_timings_mutag) +
                      ['Assignment'] * len(assignment_timings_mutag) +
                      ['Parallel'] * len(parallel_timings_mutag) +
                      ['LSA'] * len(lsap_timings_ohsu) +
                      ['Divider'] * len(divider_timings_ohsu) +
                      ['Assignment'] * len(assignment_timings_ohsu) +
                      ['Parallel'] * len(parallel_timings_ohsu)),
        'Dataset': (['Mutag'] * (
                len(lsap_timings_mutag) + len(assignment_timings_mutag) + len(divider_timings_mutag) + len(
            parallel_timings_mutag)) +
                    ['OHSU'] * (len(lsap_timings_ohsu) + len(assignment_timings_ohsu) + len(divider_timings_ohsu) + len(
                    parallel_timings_ohsu)))
    }

    df = pd.DataFrame(data)

    # Create a 'Group' column: e.g., 'LSAP-Mutag'
    df['Group'] = df['Algorithm'] + '-' + df['Dataset']

    # ----------- Boxplot -----------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Group', y='Time', hue='Dataset', data=df, dodge=False,
                palette={'Mutag': '#1f77b4', 'OHSU': '#ff7f0e'})
    plt.title(f'Algorithm Timings per Dataset (Boxplot) for {runs} runs')
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')
    plt.show()

    data = []
    for run_idx, run in enumerate(divider_true_positives_mutag):
        for tp in run:
            data.append({
                'Run': f'Run {run_idx + 1}',
                'TruePositivePercent': tp,
                'Algorithm': 'Divider'
            })
    for run_idx, run in enumerate(assign_true_positives_mutag):
        for tp in run:
            data.append({
                'Run': f'Run {run_idx + 1}',
                'TruePositivePercent': tp,
                'Algorithm': 'Assignment'
            })
    for run_idx, run in enumerate(parallel_true_positives_mutag):
        for tp in run:
            data.append({
                'Run': f'Run {run_idx + 1}',
                'TruePositivePercent': tp,
                'Algorithm': 'Parallel'
            })

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Run', y='TruePositivePercent', hue='Algorithm', data=df)
    plt.xticks(rotation=45)
    plt.ylabel('True Positive Percentage (%)')
    plt.title('TP% Distribution per Run: Greedy vs Assignment MUTAG')
    plt.ylim(None, 105)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')

    plt.show()

    data_ohsu = []

    for run_idx, run in enumerate(divider_true_positives_ohsu):
        for tp in run:
            data_ohsu.append({
                'Run': f'Run {run_idx + 1}',
                'TruePositivePercent': tp,
                'Algorithm': 'Divider'
            })
    for run_idx, run in enumerate(assign_true_positives_ohsu):
        for tp in run:
            data_ohsu.append({
                'Run': f'Run {run_idx + 1}',
                'TruePositivePercent': tp,
                'Algorithm': 'Assignment'
            })
    for run_idx, run in enumerate(parallel_true_positives_ohsu):
        for tp in run:
            data_ohsu.append({
                'Run': f'Run {run_idx + 1}',
                'TruePositivePercent': tp,
                'Algorithm': 'Parallel'
            })

    df_ohsu = pd.DataFrame(data_ohsu)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Run', y='TruePositivePercent', hue='Algorithm', data=df_ohsu)
    plt.xticks(rotation=45)
    plt.ylabel('True Positive Percentage (%)')
    plt.title('TP% Distribution per Run: Greedy vs Assignment OSHU')
    plt.ylim(None, 105)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'./png/{timestamp}.png')

    plt.show()



    def compute_error_df(lsap_list, compare_list, dataset_name, method_name):
        error_data = []
        for run_idx in range(len(lsap_list)):
            for matrix_idx in range(len(lsap_list[run_idx])):
                lsa_cost = lsap_list[run_idx][matrix_idx]
                compare_cost = compare_list[run_idx][matrix_idx]
                sq_error = (compare_cost - lsa_cost) ** 2
                error_data.append({
                    "Dataset": dataset_name,
                    "Run": run_idx,
                    "Method": method_name,
                    "Squared Error": sq_error,
                    "True Cost": lsa_cost
                })
        return pd.DataFrame(error_data)


    # Compute errors for both parallel and assignment against LSA for MUTAG
    df_mutag_parallel = compute_error_df(lsap_minimal_sum_mutag, parallel_minimal_sum_mutag, "MUTAG", "Parallel")
    df_mutag_assignment = compute_error_df(lsap_minimal_sum_mutag, assignment_minimal_sum_mutag, "MUTAG", "Assignment")
    df_mutag_divider = compute_error_df(lsap_minimal_sum_mutag, divider_minimal_sum_mutag, "MUTAG", "Divider")

    # Compute errors for both parallel and assignment against LSA for OHSU
    df_ohsu_parallel = compute_error_df(lsap_minimal_sum_ohsu, parallel_minimal_sum_ohsu, "OHSU", "Parallel")
    df_ohsu_assignment = compute_error_df(lsap_minimal_sum_ohsu, assignment_minimal_sum_ohsu, "OHSU", "Assignment")
    df_ohsu_divider = compute_error_df(lsap_minimal_sum_ohsu, divider_minimal_sum_ohsu, "OHSU", "Divider")

    # Combine all data
    error_df = pd.concat(
        [df_mutag_parallel, df_mutag_assignment, df_mutag_divider, df_ohsu_parallel, df_ohsu_assignment,
         df_ohsu_divider],
        ignore_index=True)

    # Group by Dataset and Method to get RMSE and pRMSE
    grouped = error_df.groupby(["Dataset", "Method"])
    rmse = grouped["Squared Error"].mean().apply(np.sqrt)
    mean_true_cost = grouped["True Cost"].mean()
    pRMSE = (rmse / mean_true_cost) * 100

    # Reshape for plotting
    pRMSE_df = pRMSE.reset_index()

    # Plot grouped bar chart

    plt.figure(figsize=(10, 6))
    sns.barplot(data=pRMSE_df, x="Dataset", y=0, hue="Method", palette=["#0072B2", "#D55E00", "#009E73"])
    plt.ylabel("Percent RMSE (%)")
    plt.title("Percent RMSE Between LSA and Parallel/Assignment Methods (by Dataset)")
    plt.grid(axis='y')
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'./png/{timestamp}.png')
    plt.show()


    def scatter_lsa_vs_algo(lsa_list, algo_list, dataset_name, algo_name):
        # Flatten all runs and matrices into 1D arrays for plotting
        lsa_flat = [cost for run in lsa_list for cost in run]
        algo_flat = [cost for run in algo_list for cost in run]

        # Compute limits for x and y axes
        combined_min = min(min(lsa_flat), min(algo_flat))
        combined_max = max(max(lsa_flat), max(algo_flat))

        plt.figure(figsize=(7, 7))
        plt.scatter(lsa_flat, algo_flat, alpha=0.6, edgecolor='k', s=40)
        plt.plot([combined_min, combined_max], [combined_min, combined_max], 'r--', label='y = x')
        plt.xlabel('LSA Cost')
        plt.ylabel(f'{algo_name} Cost')
        plt.title(f'Scatter Plot: LSA vs {algo_name} ({dataset_name})')
        plt.xlim(combined_min, combined_max)
        plt.ylim(combined_min, combined_max)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'./png/{timestamp}_{algo_name}.png'
        plt.savefig(filename)
        plt.show()


    # For OHSU dataset
    scatter_lsa_vs_algo(lsap_minimal_sum_ohsu, assignment_minimal_sum_ohsu, "OHSU", "Assignment")
    scatter_lsa_vs_algo(lsap_minimal_sum_ohsu, divider_minimal_sum_ohsu, "OHSU", "Divider")
    scatter_lsa_vs_algo(lsap_minimal_sum_ohsu, parallel_minimal_sum_ohsu, "OHSU", "Parallel")

    # For MUTAG dataset
    scatter_lsa_vs_algo(lsap_minimal_sum_mutag, assignment_minimal_sum_mutag, "MUTAG", "Assignment")
    scatter_lsa_vs_algo(lsap_minimal_sum_mutag, divider_minimal_sum_mutag, "MUTAG", "Divider")
    scatter_lsa_vs_algo(lsap_minimal_sum_mutag, parallel_minimal_sum_mutag, "MUTAG", "Parallel")
