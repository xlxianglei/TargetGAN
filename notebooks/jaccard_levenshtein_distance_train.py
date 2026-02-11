import numpy as np
import pandas as pd
import random
import Levenshtein
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm  # Highly recommended library: pip install tqdm, used for progress bars

# --- 1. Configuration and Helper Functions ---
seed = 2024
random.seed(seed)
num = 640
k_mer = 4
# Note: The filename format must match your actual path
itera = ['100','1000','5000', '10000','20000','54000','80000','100000'] 
base_path = '../wgan-gp/z_dim_100_gen_dim_200_disc_dim_200/samples'

def jaccard_distance(seq1, seq2, k=4):
    """Calculate the Jaccard distance between two sequences"""
    # Use generator expressions instead of list comprehensions to save memory
    set1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1))
    set2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return 1 - intersection / union

def calculate_single_seq_metrics(query_seq, train_seqs):
    """
    Calculate the minimum distance between a single query sequence and the entire training set.
    This function will be called in parallel.
    """
    # 1. Calculate Jaccard distance to all training data, take the minimum
    # Pass generator expression to min() to avoid generating huge intermediate lists
    min_j = min(jaccard_distance(query_seq, t) for t in train_seqs)
    
    # 2. Calculate Levenshtein distance to all training data, take the minimum
    min_l = min(Levenshtein.distance(query_seq, t) for t in train_seqs)
    
    return min_j, min_l

def main():
    print("Reading data...")
    natural_path = '../data/Natural_promoters.xlsx'
    df_nat = pd.read_excel(natural_path)
    
    # Extract training set and convert to list for faster reading in multiprocessing
    train_seq = list(df_nat[df_nat.dataset == 'training set'].sequence)
    data_test = df_nat[df_nat.dataset == 'test set']
    
    print(f"Training set size: {len(train_seq)}")
    
    # Initialize result DataFrame
    jaccard_df = pd.DataFrame()
    Levenshtein_df = pd.DataFrame()

    # --- 2. Core Calculation Loop (Parallelized) ---
    
    # Automatically adjust max_workers based on CPU cores; usually setting to None uses all cores
    # If on a shared server, it's recommended to manually set to 10-20
    max_workers = 20 
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        # Use partial to fix the train_seq parameter, so map only needs to pass query_seq
        # Note: On Linux utilizing the fork mechanism, this step has very low overhead; might be slower on Windows
        worker_func = partial(calculate_single_seq_metrics, train_seqs=train_seq)

        # A. Process generated sequences (Itera)
        for n in itera:
            file_path = os.path.join(base_path, f'samples_{n}')
            print(f"Processing Iteration: {n} ...")
            
            try:
                with open(file_path, 'r') as f:
                    # Read and remove newline characters
                    gen_seqs = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found, skipping.")
                continue

            # Parallel calculation core step
            # executor.map returns results in order
            # list() triggers the actual calculation
            results = list(tqdm(executor.map(worker_func, gen_seqs), 
                                total=len(gen_seqs), desc=f"Iter {n}"))
            
            # Unpack results [(j1, l1), (j2, l2)...] -> [j1, j2...], [l1, l2...]
            j_scores, l_scores = zip(*results)
            
            jaccard_df[str(n)] = j_scores
            Levenshtein_df[str(n)] = l_scores

        # B. Process Test Set
        print("Processing Test Set ...")
        test_seq = random.sample(list(data_test.sequence), num)
        
        results_test = list(tqdm(executor.map(worker_func, test_seq), 
                                 total=len(test_seq), desc="Test Set"))
        
        test_j, test_l = zip(*results_test)
        jaccard_df['test'] = test_j
        Levenshtein_df['test'] = test_l

    # --- 3. Save Results ---
    print("Calculation complete, saving results...")
    jaccard_df.to_csv('./data_for_plot/result_min_Jaccard.csv', index=False)
    Levenshtein_df.to_csv('./data_for_plot/result_min_Levenshtein.csv', index=False) # can also save as Excel
    
    print("Results saved to result_min_Jaccard.csv and result_min_Levenshtein.csv")
    
    # Briefly print a preview
    print("\nJaccard DataFrame Head:")
    print(jaccard_df.head())

if __name__ == '__main__':
    main()