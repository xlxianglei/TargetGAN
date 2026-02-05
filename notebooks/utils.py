import os
import logging
import numpy as np
from functools import reduce
from typing import Any

def suppress_tf_warnings():
    """Suppress TensorFlow logs and warnings."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=All, 1=Filter INFO, 2=Filter INFO & WARN, 3=Filter All
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

def kmer_frequencies(seq, k):
    kmer_all = reduce(lambda x,y: [i+j for i in x for j in y], [['A','T','C','G']] * k)
    kmer_freq = dict.fromkeys(kmer_all, 0)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmer_freq[kmer] = kmer_freq.get(kmer, 0) + 1
    return kmer_freq

def accumulate_kmer_frequencies(sequences, k):
    accumulated_freq = {}
    for seq in sequences:
        kmer_freq = kmer_frequencies(seq, k)
        for kmer, freq in kmer_freq.items():
            accumulated_freq[kmer] = accumulated_freq.get(kmer, 0) + freq
    accumulated_vector = np.array(list(accumulated_freq.values()))/sum(list(accumulated_freq.values()))
    return accumulated_vector

def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float64) -> np.ndarray:
    """One-hot encode sequence."""
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]

def jaccard_distance(seq1, seq2, k: int = 4) -> float:
    """Compute Jaccard distance between two DNA sequences."""
    set1 = set(seq1[i:i+k] for i in range(len(seq1)-k+1))
    set2 = set(seq2[i:i+k] for i in range(len(seq2)-k+1))
    intersection = set1 & set2
    union = set1 | set2
    return 1 - len(intersection) / len(union)

def gc_content(x):
    """Compute GC content of a DNA sequence."""
    return np.mean([int(xx in ['G','C']) for xx in x])

def load_dna_shape_means(filepath):
    """
    Reads DNA shape data from a file.
    Assumes FASTA-like format or blocks separated by headers starting with '>'.
    Calculates the mean value for each sequence.
    """
    means = []
    current_values = []
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return np.array([])

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('>'):
                # End of previous sequence
                if current_values:
                    means.append(np.mean(current_values))
                    current_values = []
                # New sequence starts (header line)
            else:
                # Data line
                parts = line.split(',')
                for x in parts:
                    x = x.strip()
                    if x and x != 'NA':
                        try:
                            current_values.append(float(x))
                        except ValueError:
                            pass # Handle non-numeric if any
                            
        # Append the last sequence
        if current_values:
            means.append(np.mean(current_values))
            
    return np.array(means)
def reversed_seq(seqs):
    """
    Generates the reverse complement of a list of DNA sequences.
    """
    # Complement mapping
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    revers_seq = []
    # Loop through sequences
    for i in seqs:
        # Reverse the sequence
        reversed_sequence = i[::-1]
        # Generate complement
        complement_sequence = ''.join([complement[base] for base in reversed_sequence])
        revers_seq.append(complement_sequence)
    return revers_seq

def get_median_row(group):
    """
    Selects the row corresponding to the median enrichment value.
    If 'reversed_bar' is not unique, it finds the row closest to the median enrichment.
    """
    # If there are multiple unique barcodes for the gene
    if group['reversed_bar'].nunique() > 1:
        median_val = group['enrichment'].median()
        # Find the row closest to the median (take the first one if multiple)
        median_rows = group.iloc[(group['enrichment'] - median_val).abs().argsort()[:1]]
        return median_rows
    # Otherwise return the top row
    else:
        return group.head(1)
