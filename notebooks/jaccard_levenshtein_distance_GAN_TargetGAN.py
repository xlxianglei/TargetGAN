import numpy as np
import pandas as pd
import random
import Levenshtein
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

# --- 1. Configuration Parameters ---
seed = 2024
random.seed(seed)
k_mer = 4
output_dir = './data_for_plot'  # Folder to save results

# File paths (Please confirm the file names are correct)
path_natural = '../data/Natural_promoters.xlsx'
path_wgan = '../data/WGAN_GP_gen_pro.xlsx'
path_target = '../data/targetgan_55296.xlsx'

# --- 2. Core Calculation Functions ---

def jaccard_distance(seq1, seq2, k=4):
    """Calculate Jaccard distance between two sequences (1 - IoU)"""
    if not seq1 or not seq2: return 1.0
    
    # Use generator expressions to avoid generating huge lists and save memory
    set1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1))
    set2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1))
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0: return 1.0
    return 1 - intersection / union

def calculate_single_seq_metrics(query_seq, train_seqs):
    """
    Calculate the minimum distance between a single query sequence and the entire training set.
    """
    # 1. Jaccard: Find minimum value (nearest neighbor)
    min_j = min(jaccard_distance(query_seq, t) for t in train_seqs)
    
    # 2. Levenshtein: Find minimum value (edit distance, nearest neighbor)
    min_l = min(Levenshtein.distance(query_seq, t) for t in train_seqs)
    
    return min_j, min_l

# --- 3. Main Program ---

def main():
    print(f"Reading data...")
    
    # Read data
    df_nat = pd.read_excel(path_natural)
    df_wgan = pd.read_excel(path_wgan)
    df_target = pd.read_excel(path_target)
    
    # Extract sequences
    train_seq = list(df_nat[df_nat.dataset == 'training set'].sequence)
    
    # Prepare three task groups
    tasks = {
        'test': list(df_nat[df_nat.dataset == 'test set'].sequence),
        'wgan': list(df_wgan.sequence),
        'target': list(df_target.sequence)
    }
    
    print(f"Benchmark training set size: {len(train_seq)}")
    for name, seqs in tasks.items():
        print(f"Group to calculate [{name}] size: {len(seqs)}")

    # Parallel calculation settings
    max_workers = 20  # Adjust according to your CPU cores
    
    print(f"\nStarting parallel calculation (using {max_workers} cores)...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Fix train_seq parameter
        worker_func = partial(calculate_single_seq_metrics, train_seqs=train_seq)

        for name, sequences in tasks.items():
            print(f"\nProcessing: {name} ...")
            
            # Submit tasks and show progress bar
            results = list(tqdm(executor.map(worker_func, sequences), 
                                total=len(sequences), desc=name))
            
            # Unpack results: list of tuples -> tuple of lists
            j_scores, l_scores = zip(*results)
            
            # Convert to numpy arrays
            np_j = np.array(j_scores)
            np_l = np.array(l_scores)
            
            # --- Save as .npy files ---
            file_j = os.path.join(output_dir, f'{name}_jaccard.npy')
            file_l = os.path.join(output_dir, f'{name}_levenshtein.npy')
            
            np.save(file_j, np_j)
            np.save(file_l, np_l)
            
            print(f"  -> Saved: {file_j}")
            print(f"  -> Saved: {file_l}")

    print("\nAll calculations completed!")

if __name__ == '__main__':
    main()