"""
Timing Evaluation for Real World Datasets
Evaluates LSA, Divider, Bucket, and Direct algorithms on MUTAG, OHSU, and Proteins datasets
with multiple timing runs for statistical validity
"""

import time
import datetime
import gc
import numpy as np
import pandas as pd
from itertools import combinations

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

# Dataset configuration
len_mutag = 188
len_ohsu = 79
number_of_matrices_mutag = len(list(combinations(range(len_mutag), r=2)))
number_of_matrices_ohsu = len(list(combinations(range(len_ohsu), r=2)))
number_of_matrices_proteins = 10000  # Subset - full would be ~600,000 matrices

number_of_runs = 10  # Number of timing iterations for statistical validity

def load_datasets():
    """Load all real-world datasets"""
    print("Loading real-world datasets...")
    
    try:
        cost_matrices_mutag = MockDataGenerator.load_h5_file('./data/cost_matrices.h5', number_of_matrices_mutag)
        print(f"✓ MUTAG dataset loaded: {number_of_matrices_mutag} matrices")
    except Exception as e:
        print(f"✗ Error loading MUTAG: {e}")
        cost_matrices_mutag = None
    
    try:
        cost_matrices_ohsu = MockDataGenerator.load_h5_file('./data/cost_matrices_OHSU.h5', number_of_matrices_ohsu)
        print(f"✓ OHSU dataset loaded: {number_of_matrices_ohsu} matrices")
    except Exception as e:
        print(f"✗ Error loading OHSU: {e}")
        cost_matrices_ohsu = None
    
    try:
        cost_matrices_proteins = MockDataGenerator.load_h5_file('./data/cost_matrices_PROTEINS_subset.h5', number_of_matrices_proteins)
        print(f"✓ Proteins subset loaded: {number_of_matrices_proteins} matrices")
        print("  Note: Using subset to avoid 600,000+ matrix computation")
    except Exception as e:
        print(f"✗ Error loading Proteins: {e}")
        cost_matrices_proteins = None
    
    return cost_matrices_mutag, cost_matrices_ohsu, cost_matrices_proteins

def measure_algorithm_timing(cost_matrices, algorithm_name, dataset_name, num_matrices):
    """Measure timing for a specific algorithm on given cost matrices"""
    if cost_matrices is None:
        return None
        
    start_time = time.time()
    
    try:
        if algorithm_name == 'lsa':
            results = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) 
                      for i in range(num_matrices)]
        elif algorithm_name == 'divider':
            results = MatrixDivider.divider(cost_matrices, num_matrices, 4)
        elif algorithm_name == 'bucket':
            results = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
        elif algorithm_name == 'direct':
            results = [AssignmentAlgo.assignment_applied(cost_matrices[i]) 
                      for i in range(num_matrices)]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Clean up results to free memory
        del results
        gc.collect()
        
        return execution_time
        
    except Exception as e:
        print(f"  ✗ Error in {algorithm_name} for {dataset_name}: {e}")
        return None

def run_timing_evaluation():
    """Run comprehensive timing evaluation for all datasets and algorithms"""
    print("=" * 80)
    print("REAL WORLD DATASETS TIMING EVALUATION")
    print("=" * 80)
    print(f"Number of timing runs per dataset: {number_of_runs}")
    print()
    
    # Load datasets
    datasets = load_datasets()
    cost_matrices_mutag, cost_matrices_ohsu, cost_matrices_proteins = datasets
    
    # Dataset configuration
    dataset_config = [
        ('MUTAG', cost_matrices_mutag, number_of_matrices_mutag),
        ('OHSU', cost_matrices_ohsu, number_of_matrices_ohsu), 
        ('Proteins', cost_matrices_proteins, number_of_matrices_proteins)
    ]
    
    algorithms = ['lsa', 'divider', 'bucket', 'direct']
    timing_results = []
    
    total_start = time.time()
    
    # Run timing evaluation
    for run_idx in range(number_of_runs):
        print(f"\\nRun {run_idx + 1}/{number_of_runs}")
        print("-" * 40)
        
        run_start = time.time()
        
        for dataset_name, cost_matrices, num_matrices in dataset_config:
            if cost_matrices is None:
                print(f"  Skipping {dataset_name} - dataset not available")
                continue
                
            print(f"  Processing {dataset_name} ({num_matrices} matrices)")
            
            run_timings = {
                'run': run_idx + 1,
                'dataset': dataset_name,
                'num_matrices': num_matrices
            }
            
            # Measure each algorithm
            for algo in algorithms:
                print(f"    {algo.upper()}...", end=" ")
                timing = measure_algorithm_timing(cost_matrices, algo, dataset_name, num_matrices)
                run_timings[f'{algo}_time'] = round(timing, 4) if timing is not None else None
                if timing is not None:
                    print(f"{timing:.4f}s")
                else:
                    print("FAILED")
            
            timing_results.append(run_timings)
        
        run_end = time.time()
        print(f"  Run completed in {run_end - run_start:.2f} seconds")
    
    total_end = time.time()
    print(f"\\nTotal evaluation time: {total_end - total_start:.2f} seconds")
    
    return timing_results

def save_timing_results(timing_results):
    """Save timing results to CSV files"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Create DataFrame
    df = pd.DataFrame(timing_results)
    
    # Save detailed results
    detailed_filename = f'./csv/timing_real_datasets_detailed_{timestamp}.csv'
    df.to_csv(detailed_filename, sep=';', index=False)
    print(f'\\nDetailed timing results saved to: {detailed_filename}')
    
    # Create summary statistics
    summary_data = []
    datasets = df['dataset'].unique()
    algorithms = ['lsa', 'divider', 'bucket', 'direct']
    
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        
        if len(dataset_data) > 0:
            summary_row = {
                'dataset': dataset,
                'num_matrices': dataset_data['num_matrices'].iloc[0]
            }
            
            for algo in algorithms:
                algo_times = dataset_data[f'{algo}_time'].dropna()
                if len(algo_times) > 0:
                    summary_row.update({
                        f'{algo}_mean': round(algo_times.mean(), 4),
                        f'{algo}_std': round(algo_times.std(), 4),
                        f'{algo}_min': round(algo_times.min(), 4),
                        f'{algo}_max': round(algo_times.max(), 4),
                        f'{algo}_runs': len(algo_times)
                    })
                else:
                    summary_row.update({
                        f'{algo}_mean': None,
                        f'{algo}_std': None,
                        f'{algo}_min': None,
                        f'{algo}_max': None,
                        f'{algo}_runs': 0
                    })
            
            summary_data.append(summary_row)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'./csv/timing_real_datasets_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, sep=';', index=False)
    print(f'Summary timing results saved to: {summary_filename}')
    
    # Display summary
    print("\\nTiming Summary:")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['dataset']} Dataset ({row['num_matrices']} matrices):")
        for algo in algorithms:
            if row[f'{algo}_runs'] > 0:
                print(f"  {algo.upper()}: {row[f'{algo}_mean']:.4f}s ± {row[f'{algo}_std']:.4f}s "
                      f"(min: {row[f'{algo}_min']:.4f}s, max: {row[f'{algo}_max']:.4f}s)")
            else:
                print(f"  {algo.upper()}: No successful runs")
        print()
    
    return detailed_filename, summary_filename

if __name__ == '__main__':
    print("REAL WORLD DATASETS - TIMING EVALUATION")
    print()
    print("Datasets:")
    print(f"  • MUTAG: {number_of_matrices_mutag} matrices")
    print(f"  • OHSU: {number_of_matrices_ohsu} matrices") 
    print(f"  • Proteins: {number_of_matrices_proteins} matrices (subset)")
    print()
    print("Note: Proteins dataset uses subset to avoid 600,000+ matrix computation")
    print("      which would take ~300 seconds per algorithm per run.")
    print()
    
    # Run the evaluation
    timing_results = run_timing_evaluation()
    
    # Save results
    if timing_results:
        detailed_file, summary_file = save_timing_results(timing_results)
        
        print("=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total measurements: {len(timing_results)}")
        print(f"Detailed results: {detailed_file}")
        print(f"Summary results: {summary_file}")
    else:
        print("\\nNo timing results to save!")