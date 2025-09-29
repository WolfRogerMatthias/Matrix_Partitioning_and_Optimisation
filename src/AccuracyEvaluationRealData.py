"""
Accuracy Evaluation for Real World Datasets
Evaluates accuracy of Divider, Bucket, and Direct algorithms compared to optimal LSA
on MUTAG, OHSU, and Proteins datasets
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

def calculate_accuracy(optimal_mapping, algorithm_mapping, algorithm_name):
    """Calculate assignment accuracy compared to optimal solution"""
    accuracy_list = []
    
    for i in range(len(optimal_mapping)):
        optimal_row_ind, optimal_col_ind = optimal_mapping[i]
        optimal_assignments = set(zip(optimal_row_ind, optimal_col_ind))
        
        if algorithm_name == 'divider':
            algo_row_ind, algo_col_ind = algorithm_mapping[i]
            algo_assignments = set(zip(algo_row_ind, algo_col_ind))
        elif algorithm_name == 'bucket':
            algo_assignments = set(algorithm_mapping[i])
        elif algorithm_name == 'direct':
            algo_assignments = set(algorithm_mapping[i].items())
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Calculate number of correct assignments
        correct_assignments = len(optimal_assignments.intersection(algo_assignments))
        total_assignments = len(optimal_assignments)
        accuracy_percentage = (correct_assignments / total_assignments) * 100 if total_assignments > 0 else 0
        accuracy_list.append(accuracy_percentage)
    
    return np.array(accuracy_list)

def calculate_minimal_sum(cost_matrices, algorithm_mapping, algorithm_name):
    """Calculate minimal sum for the algorithm"""
    min_sum_list = []
    
    for i in range(len(cost_matrices)):
        matrix = cost_matrices[i]
        
        # Algorithm cost
        if algorithm_name == 'lsa':
            row_ind, col_ind = algorithm_mapping[i]
            min_sum = matrix[row_ind, col_ind].sum()
        elif algorithm_name == 'divider':
            row_ind, col_ind = algorithm_mapping[i]
            min_sum = matrix[row_ind, col_ind].sum()
        elif algorithm_name == 'bucket':
            min_sum = sum(matrix[x, y] for x, y in algorithm_mapping[i])
        elif algorithm_name == 'direct':
            min_sum = sum(matrix[x, y] for x, y in algorithm_mapping[i].items())
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        min_sum_list.append(min_sum)
    
    return np.array(min_sum_list)

def process_dataset(dataset_name, cost_matrices, num_matrices):
    """Process a single dataset to compute all algorithm results"""
    if cost_matrices is None:
        return None
        
    print(f"\\nProcessing {dataset_name} dataset ({num_matrices} matrices)")
    
    try:
        # Compute optimal LSA mapping
        print("  Computing LSA (optimal)...")
        start_time = time.time()
        lsa_mapping = [OptimizeAlgoApplied.compute_linear_sum_assignment(cost_matrices[i]) 
                      for i in range(num_matrices)]
        print(f"    LSA completed in {time.time() - start_time:.2f} seconds")
        
        # Compute algorithm mappings
        print("  Computing Divider...")
        start_time = time.time()
        divider_mapping = MatrixDivider.divider(cost_matrices, num_matrices, 4)
        print(f"    Divider completed in {time.time() - start_time:.2f} seconds")
        
        print("  Computing Bucket...")
        start_time = time.time()
        bucket_mapping = ParallelBucketAssignmentSolver.solve_multiple(cost_matrices, 2)
        print(f"    Bucket completed in {time.time() - start_time:.2f} seconds")
        
        print("  Computing Direct...")
        start_time = time.time()
        direct_mapping = [AssignmentAlgo.assignment_applied(cost_matrices[i]) 
                         for i in range(num_matrices)]
        print(f"    Direct completed in {time.time() - start_time:.2f} seconds")
        
        # Calculate accuracies
        print("  Calculating accuracies...")
        divider_accuracy = calculate_accuracy(lsa_mapping, divider_mapping, 'divider')
        bucket_accuracy = calculate_accuracy(lsa_mapping, bucket_mapping, 'bucket') 
        direct_accuracy = calculate_accuracy(lsa_mapping, direct_mapping, 'direct')
        
        # Calculate minimal sums
        print("  Calculating minimal sums...")
        lsa_min_sum = calculate_minimal_sum(cost_matrices, lsa_mapping, 'lsa')
        divider_min_sum = calculate_minimal_sum(cost_matrices, divider_mapping, 'divider')
        bucket_min_sum = calculate_minimal_sum(cost_matrices, bucket_mapping, 'bucket')
        direct_min_sum = calculate_minimal_sum(cost_matrices, direct_mapping, 'direct')
        
        # Clean up memory
        del lsa_mapping, divider_mapping, bucket_mapping, direct_mapping
        gc.collect()
        
        return {
            'dataset': dataset_name,
            'num_matrices': num_matrices,
            'divider_accuracy': divider_accuracy,
            'bucket_accuracy': bucket_accuracy,
            'direct_accuracy': direct_accuracy,
            'lsa_min_sum': lsa_min_sum,
            'divider_min_sum': divider_min_sum,
            'bucket_min_sum': bucket_min_sum,
            'direct_min_sum': direct_min_sum
        }
        
    except Exception as e:
        print(f"  ✗ Error processing {dataset_name}: {e}")
        return None

def run_accuracy_evaluation():
    """Run comprehensive accuracy evaluation for all datasets"""
    print("=" * 80)
    print("REAL WORLD DATASETS ACCURACY EVALUATION")
    print("=" * 80)
    
    # Load datasets
    datasets = load_datasets()
    cost_matrices_mutag, cost_matrices_ohsu, cost_matrices_proteins = datasets
    
    # Dataset configuration
    dataset_config = [
        ('MUTAG', cost_matrices_mutag, number_of_matrices_mutag),
        ('OHSU', cost_matrices_ohsu, number_of_matrices_ohsu), 
        ('Proteins', cost_matrices_proteins, number_of_matrices_proteins)
    ]
    
    results = []
    total_start = time.time()
    
    # Process each dataset
    for dataset_name, cost_matrices, num_matrices in dataset_config:
        result = process_dataset(dataset_name, cost_matrices, num_matrices)
        if result is not None:
            results.append(result)
    
    total_end = time.time()
    print(f"\\nTotal evaluation time: {total_end - total_start:.2f} seconds")
    
    return results

def save_accuracy_results(results):
    """Save accuracy and minimal sum results to CSV files"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Prepare detailed data for CSV export
    detailed_accuracy_data = []
    detailed_min_sum_data = []
    
    for result in results:
        dataset_name = result['dataset']
        num_matrices = result['num_matrices']
        
        # Create detailed records for each matrix
        for i in range(num_matrices):
            # Accuracy data
            detailed_accuracy_data.append({
                'dataset': dataset_name,
                'matrix_id': i + 1,
                'divider_accuracy': round(result['divider_accuracy'][i], 2),
                'bucket_accuracy': round(result['bucket_accuracy'][i], 2),
                'direct_accuracy': round(result['direct_accuracy'][i], 2)
            })
            
            # Minimal sum data
            detailed_min_sum_data.append({
                'dataset': dataset_name,
                'matrix_id': i + 1,
                'lsa_min_sum': round(result['lsa_min_sum'][i], 2),
                'divider_min_sum': round(result['divider_min_sum'][i], 2),
                'bucket_min_sum': round(result['bucket_min_sum'][i], 2),
                'direct_min_sum': round(result['direct_min_sum'][i], 2)
            })
    
    # Save detailed accuracy results
    accuracy_df = pd.DataFrame(detailed_accuracy_data)
    accuracy_filename = f'./csv/acc_real_data_{timestamp}.csv'
    accuracy_df.to_csv(accuracy_filename, sep=';', index=False)
    print(f'\\nDetailed accuracy results saved to: {accuracy_filename}')
    
    # Save detailed minimal sum results
    min_sum_df = pd.DataFrame(detailed_min_sum_data)
    min_sum_filename = f'./csv/cost_real_data_{timestamp}.csv'
    min_sum_df.to_csv(min_sum_filename, sep=';', index=False)
    print(f'Detailed minimal sum results saved to: {min_sum_filename}')
    
    # Create summary statistics
    summary_data = []
    algorithms = ['lsa', 'divider', 'bucket', 'direct']
    accuracy_algorithms = ['divider', 'bucket', 'direct']
    
    for result in results:
        dataset_name = result['dataset']
        num_matrices = result['num_matrices']
        
        summary_row = {
            'dataset': dataset_name,
            'num_matrices': num_matrices
        }
        
        # Accuracy statistics
        for algo in accuracy_algorithms:
            accuracy_data = result[f'{algo}_accuracy']
            summary_row.update({
                f'{algo}_accuracy_mean': round(np.mean(accuracy_data), 2),
                f'{algo}_accuracy_std': round(np.std(accuracy_data), 2),
                f'{algo}_accuracy_median': round(np.median(accuracy_data), 2),
                f'{algo}_accuracy_min': round(np.min(accuracy_data), 2),
                f'{algo}_accuracy_max': round(np.max(accuracy_data), 2)
            })
        
        # Minimal sum statistics  
        for algo in algorithms:
            min_sum_data = result[f'{algo}_min_sum']
            summary_row.update({
                f'{algo}_min_sum_mean': round(np.mean(min_sum_data), 2),
                f'{algo}_min_sum_std': round(np.std(min_sum_data), 2),
                f'{algo}_min_sum_median': round(np.median(min_sum_data), 2),
                f'{algo}_min_sum_min': round(np.min(min_sum_data), 2),
                f'{algo}_min_sum_max': round(np.max(min_sum_data), 2)
            })
        
        summary_data.append(summary_row)
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f'./csv/real_data_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, sep=';', index=False)
    print(f'Summary results saved to: {summary_filename}')
    
    # Display summary
    print("\\nAccuracy & Minimal Sum Summary:")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['dataset']} Dataset ({row['num_matrices']} matrices):")
        print("  Accuracy (%):")
        for algo in accuracy_algorithms:
            print(f"    {algo.capitalize()}: {row[f'{algo}_accuracy_mean']:.2f}% ± {row[f'{algo}_accuracy_std']:.2f}% "
                  f"(median: {row[f'{algo}_accuracy_median']:.2f}%)")
        print("  Minimal Sum:")
        for algo in algorithms:
            print(f"    {algo.capitalize()}: {row[f'{algo}_min_sum_mean']:.2f} ± {row[f'{algo}_min_sum_std']:.2f} "
                  f"(median: {row[f'{algo}_min_sum_median']:.2f})")
        print()
    
    return accuracy_filename, min_sum_filename, summary_filename

if __name__ == '__main__':
    print("REAL WORLD DATASETS - ACCURACY EVALUATION")
    print()
    print("Datasets:")
    print(f"  • MUTAG: {number_of_matrices_mutag} matrices")
    print(f"  • OHSU: {number_of_matrices_ohsu} matrices") 
    print(f"  • Proteins: {number_of_matrices_proteins} matrices (subset)")
    print()
    print("Note: Proteins dataset uses subset to avoid 600,000+ matrix computation")
    print("      which would take extensive processing time.")
    print()
    
    # Run the evaluation
    results = run_accuracy_evaluation()
    
    # Save results
    if results:
        accuracy_file, min_sum_file, summary_file = save_accuracy_results(results)
        
        print("=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Datasets processed: {len(results)}")
        print(f"Accuracy results: {accuracy_file}")
        print(f"Minimal sum results: {min_sum_file}")
        print(f"Summary results: {summary_file}")
    else:
        print("\\nNo results to save!")