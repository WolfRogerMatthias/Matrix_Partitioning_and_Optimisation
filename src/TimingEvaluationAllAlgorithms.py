"""
Comprehensive Timing Evaluation for All Algorithms
Evaluates LSA, Divider, Bucket, and Direct algorithms across multiple matrix sizes and iterations
"""

import time
import datetime
import gc
import numpy as np
import pandas as pd

from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.MatrixDivider import MatrixDivider
from src.ParallelBucketAssignmentSolver import ParallelBucketAssignmentSolver
from src.AssignmentAlgo import AssignmentAlgo

# Initialize algorithms
MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
MatrixDivider = MatrixDivider()
ParallelBucketAssignmentSolver = ParallelBucketAssignmentSolver()
AssignmentAlgo = AssignmentAlgo()

# Configuration
matrix_sizes = [i for i in range(10, 401, 5)]  # 10 to 400 in steps of 5
number_of_matrices = 1000
number_of_iterations = 10  # Number of timing runs for each matrix size

def measure_algorithm_timing(cost_matrices, algorithm_name):
    """
    Measure timing for a specific algorithm on given cost matrices
    """
    start_time = time.time()
    
    try:
        if algorithm_name == 'lsa':
            results = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) 
                      for i in range(number_of_matrices)]
        elif algorithm_name == 'divider':
            results = MatrixDivider.divider(cost_matrices, number_of_matrices, 4)
        elif algorithm_name == 'bucket':
            results = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
        elif algorithm_name == 'direct':
            results = [AssignmentAlgo.assignment_applied(cost_matrices[i]) 
                      for i in range(number_of_matrices)]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Clean up results to free memory
        del results
        gc.collect()
        
        return execution_time
        
    except Exception as e:
        print(f"Error in {algorithm_name}: {e}")
        return None

def run_timing_evaluation():
    """
    Run comprehensive timing evaluation for all algorithms
    """
    print("Starting Comprehensive Timing Evaluation")
    print(f"Matrix sizes: {len(matrix_sizes)} sizes from {matrix_sizes[0]} to {matrix_sizes[-1]}")
    print(f"Number of matrices per size: {number_of_matrices}")
    print(f"Number of iterations per matrix size: {number_of_iterations}")
    
    # Store timing results
    timing_data = []
    
    total_start = time.time()
    
    # Iterate through each matrix size
    for size_idx, matrix_size in enumerate(matrix_sizes):
        print(f"\nProcessing matrix size {matrix_size} ({size_idx + 1}/{len(matrix_sizes)})")
        
        # Load matrices for this size once
        try:
            cost_matrices = MockDataGenerator.load_h5_file(
                f'./data/cost_matrices/cost_matrices_{matrix_size}.h5',
                number_of_matrices
            )
        except Exception as e:
            print(f"Error loading matrices for size {matrix_size}: {e}")
            continue
        
        # Run multiple iterations for this matrix size
        for iteration in range(number_of_iterations):
            print(f"  Iteration {iteration + 1}/{number_of_iterations}")
            
            # Measure timing for each algorithm
            lsa_time = measure_algorithm_timing(cost_matrices, 'lsa')
            divider_time = measure_algorithm_timing(cost_matrices, 'divider')
            bucket_time = measure_algorithm_timing(cost_matrices, 'bucket')
            direct_time = measure_algorithm_timing(cost_matrices, 'direct')
            
            # Store results
            timing_data.append({
                'matrix_size': matrix_size,
                'iteration': iteration + 1,
                'lsa_time': round(lsa_time, 4) if lsa_time is not None else None,
                'divider_time': round(divider_time, 4) if divider_time is not None else None,
                'bucket_time': round(bucket_time, 4) if bucket_time is not None else None,
                'direct_time': round(direct_time, 4) if direct_time is not None else None,
                'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            })
            
            print(f"    LSA: {lsa_time:.4f}s, Divider: {divider_time:.4f}s, "
                  f"Bucket: {bucket_time:.4f}s, Direct: {direct_time:.4f}s")
        
        # Clean up matrices after all iterations for this size
        del cost_matrices
        gc.collect()
    
    total_end = time.time()
    print(f"\nTotal evaluation time: {total_end - total_start:.2f} seconds")
    
    return timing_data

def save_timing_results(timing_data):
    """
    Save timing results to CSV files
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Create DataFrame
    df = pd.DataFrame(timing_data)
    
    # Save detailed results
    detailed_filename = f'./csv/timing_evaluation_detailed_{timestamp}.csv'
    df.to_csv(detailed_filename, sep=';', index=False)
    print(f'Detailed timing results saved to: {detailed_filename}')
    
    # Create summary statistics
    summary_data = []
    for matrix_size in matrix_sizes:
        size_data = df[df['matrix_size'] == matrix_size]
        
        if len(size_data) > 0:
            summary_data.append({
                'matrix_size': matrix_size,
                'lsa_mean': round(size_data['lsa_time'].mean(), 4),
                'lsa_std': round(size_data['lsa_time'].std(), 4),
                'lsa_min': round(size_data['lsa_time'].min(), 4),
                'lsa_max': round(size_data['lsa_time'].max(), 4),
                'divider_mean': round(size_data['divider_time'].mean(), 4),
                'divider_std': round(size_data['divider_time'].std(), 4),
                'divider_min': round(size_data['divider_time'].min(), 4),
                'divider_max': round(size_data['divider_time'].max(), 4),
                'bucket_mean': round(size_data['bucket_time'].mean(), 4),
                'bucket_std': round(size_data['bucket_time'].std(), 4),
                'bucket_min': round(size_data['bucket_time'].min(), 4),
                'bucket_max': round(size_data['bucket_time'].max(), 4),
                'direct_mean': round(size_data['direct_time'].mean(), 4),
                'direct_std': round(size_data['direct_time'].std(), 4),
                'direct_min': round(size_data['direct_time'].min(), 4),
                'direct_max': round(size_data['direct_time'].max(), 4),
            })
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'./csv/timing_evaluation_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, sep=';', index=False)
    print(f'Summary timing results saved to: {summary_filename}')
    
    return detailed_filename, summary_filename

if __name__ == '__main__':
    print("=" * 60)
    print("COMPREHENSIVE TIMING EVALUATION FOR ALL ALGORITHMS")
    print("=" * 60)
    
    # Run the evaluation
    timing_results = run_timing_evaluation()
    
    # Save results
    if timing_results:
        detailed_file, summary_file = save_timing_results(timing_results)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total measurements: {len(timing_results)}")
        print(f"Detailed results: {detailed_file}")
        print(f"Summary results: {summary_file}")
    else:
        print("\nNo timing results to save!")