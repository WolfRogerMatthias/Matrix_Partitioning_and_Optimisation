import time
import datetime
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.MatrixDivider import MatrixDivider
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver
from src.AssignmentAlgo import AssignmentAlgo

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
MatrixDivider = MatrixDivider()
ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()
AssignmentAlgo = AssignmentAlgo()

number_of_matrices = 1000
matrix_sizes = [i for i in range(10, 401, 5)]
datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def calculating_min_sum(matrices, algorithm_mapping, name):
    min_sum = []
    for i in range(number_of_matrices):
        if name == 'lsa':
            row_ind, col_ind = algorithm_mapping[i]
            min_sum.append(matrices[i][row_ind, col_ind].sum())
        if name == 'divider':
            row_ind, col_ind = algorithm_mapping[i]
            min_sum.append(matrices[i][row_ind, col_ind].sum())
        if name == 'bucket':
            tuples_ind = algorithm_mapping[i]
            min_sum.append(sum(matrices[i][x,y] for x, y in tuples_ind))
        if name == 'direct':
            dic_int = algorithm_mapping[i]
            min_sum.append(sum(matrices[i][x, y] for x, y in dic_int.items()))
    return np.array(min_sum)

def calculating_accuracy(optimal_mapping, algorithm_mapping, name):
    accuracy = []
    for i in range(number_of_matrices):
        optimal_row_ind, optimal_col_ind = optimal_mapping[i]
        optimal_assignments = set(zip(optimal_row_ind, optimal_col_ind))
        
        if name == 'divider':
            algo_row_ind, algo_col_ind = algorithm_mapping[i]
            algo_assignments = set(zip(algo_row_ind, algo_col_ind))
        elif name == 'bucket':
            algo_assignments = set(algorithm_mapping[i])
        elif name == 'direct':
            algo_assignments = set(algorithm_mapping[i].items())
        
        # Calculate number of correct assignments
        correct_assignments = len(optimal_assignments.intersection(algo_assignments))
        total_assignments = len(optimal_assignments)
        accuracy_percentage = (correct_assignments / total_assignments) * 100 if total_assignments > 0 else 0
        accuracy.append(accuracy_percentage)
    
    return np.array(accuracy)

def process_matrix_size(matrix_size):
    """Process a single matrix size with all algorithms"""
    try:
        print(f'Processing matrix size: {matrix_size}')
        
        # Load matrices for this size
        cost_matrices = MockDataGenerator.load_h5_file(f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                                                       number_of_matrices)
        
        # Run algorithms
        lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) for i in
                       range(len(cost_matrices))]
        divider_mapping = MatrixDivider.divider(cost_matrices, number_of_matrices, 4)
        bucket_mapping = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
        direct_mapping = [AssignmentAlgo.assignment_applied(cost_matrices[i]) for i in range(len(cost_matrices))]
        
        # Calculate costs
        lsa_costs = calculating_min_sum(cost_matrices, lsa_mapping, 'lsa')
        divider_costs = calculating_min_sum(cost_matrices, divider_mapping, 'divider')
        bucket_costs = calculating_min_sum(cost_matrices, bucket_mapping, 'bucket')
        direct_costs = calculating_min_sum(cost_matrices, direct_mapping, 'direct')
        
        # Calculate accuracies
        divider_acc = calculating_accuracy(lsa_mapping, divider_mapping, 'divider')
        bucket_acc = calculating_accuracy(lsa_mapping, bucket_mapping, 'bucket')
        direct_acc = calculating_accuracy(lsa_mapping, direct_mapping, 'direct')
        
        # Clean up memory
        del cost_matrices, lsa_mapping, divider_mapping, bucket_mapping, direct_mapping
        gc.collect()
        
        return {
            'matrix_size': matrix_size,
            'lsa_costs': lsa_costs,
            'divider_costs': divider_costs,
            'bucket_costs': bucket_costs,
            'direct_costs': direct_costs,
            'divider_acc': divider_acc,
            'bucket_acc': bucket_acc,
            'direct_acc': direct_acc
        }
        
    except Exception as e:
        print(f'Error processing matrix size {matrix_size}: {e}')
        return None

if __name__ == '__main__':
    # Initialize result containers with better memory management
    lsa_minimal_sums = []
    divider_minimal_sums = []
    bucket_minimal_sums = []
    direct_minimal_sums = []

    divider_accuracies = []
    bucket_accuracies = []
    direct_accuracies = []

    start = time.time()

    # Use parallel processing with memory management
    print(f"Starting evaluation with {mp.cpu_count()} CPU cores")
    max_workers = min(4, mp.cpu_count())  # Limit workers to prevent memory overflow

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all matrix sizes for processing
        future_to_size = {executor.submit(process_matrix_size, size): size for size in matrix_sizes}
        
        # Collect results as they complete
        for future in as_completed(future_to_size):
            result = future.result()
            if result is not None:
                # Store results in order
                matrix_size = result['matrix_size']
                size_index = matrix_sizes.index(matrix_size)
                
                # Ensure lists are long enough
                while len(lsa_minimal_sums) <= size_index:
                    lsa_minimal_sums.append(None)
                    divider_minimal_sums.append(None)
                    bucket_minimal_sums.append(None)
                    direct_minimal_sums.append(None)
                    divider_accuracies.append(None)
                    bucket_accuracies.append(None)
                    direct_accuracies.append(None)
                
                # Store results
                lsa_minimal_sums[size_index] = result['lsa_costs']
                divider_minimal_sums[size_index] = result['divider_costs']
                bucket_minimal_sums[size_index] = result['bucket_costs']
                direct_minimal_sums[size_index] = result['direct_costs']
                divider_accuracies[size_index] = result['divider_acc']
                bucket_accuracies[size_index] = result['bucket_acc']
                direct_accuracies[size_index] = result['direct_acc']

    # Clean up memory
    gc.collect()

    end = time.time()
    print(f'Time used: {end - start:2.2f} seconds')

    # Create separate CSVs for costs and accuracy
    cost_data = []
    accuracy_data = []

    for size_idx, matrix_size in enumerate(matrix_sizes):
        for matrix_idx in range(number_of_matrices):
            # Cost data CSV
            cost_data.append({
                'size': matrix_size,
                'name': f'matrix_{matrix_idx + 1}',
                'lsa_min_sum': round(lsa_minimal_sums[size_idx][matrix_idx], 2),
                'divider_min_sum': round(divider_minimal_sums[size_idx][matrix_idx], 2),
                'bucket_min_sum': round(bucket_minimal_sums[size_idx][matrix_idx], 2),
                'direct_min_sum': round(direct_minimal_sums[size_idx][matrix_idx], 2)
            })
            
            # Accuracy data CSV
            accuracy_data.append({
                'size': matrix_size,
                'name': f'matrix_{matrix_idx + 1}',
                'divider_accuracy': round(divider_accuracies[size_idx][matrix_idx], 2),
                'bucket_accuracy': round(bucket_accuracies[size_idx][matrix_idx], 2),
                'direct_accuracy': round(direct_accuracies[size_idx][matrix_idx], 2)
            })

    # Save separate CSV files
    cost_df = pd.DataFrame(cost_data)
    accuracy_df = pd.DataFrame(accuracy_data)

    cost_filename = f'./csv/evaluation_costs_{datetime}.csv'
    accuracy_filename = f'./csv/evaluation_accuracy_{datetime}.csv'

    cost_df.to_csv(cost_filename, sep=';', index=False)
    accuracy_df.to_csv(accuracy_filename, sep=';', index=False)

    print(f'Cost results saved to: {cost_filename}')
    print(f'Accuracy results saved to: {accuracy_filename}')

    # Also create combined CSV for backward compatibility
    combined_data = []
    for size_idx, matrix_size in enumerate(matrix_sizes):
        for matrix_idx in range(number_of_matrices):
            combined_data.append({
                'size': matrix_size,
                'name': f'matrix_{matrix_idx + 1}',
                'lsa_min_sum': round(lsa_minimal_sums[size_idx][matrix_idx], 2),
                'divider_min_sum': round(divider_minimal_sums[size_idx][matrix_idx], 2),
                'bucket_min_sum': round(bucket_minimal_sums[size_idx][matrix_idx], 2),
                'direct_min_sum': round(direct_minimal_sums[size_idx][matrix_idx], 2),
                'divider_accuracy': round(divider_accuracies[size_idx][matrix_idx], 2),
                'bucket_accuracy': round(bucket_accuracies[size_idx][matrix_idx], 2),
                'direct_accuracy': round(direct_accuracies[size_idx][matrix_idx], 2)
            })

    combined_df = pd.DataFrame(combined_data)
    combined_filename = f'./csv/evaluation_results_complete_{datetime}.csv'
    combined_df.to_csv(combined_filename, sep=';', index=False)
    print(f'Combined results saved to: {combined_filename}')
