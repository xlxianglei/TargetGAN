#%%
import re
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein

#%%
# Project Configuration
sample = 'SPO-P1R1'
result_path = '/home/xlxiang/last_TargetGAN/figplot_code/chapter_5_starr_seq/script/repeat_1'
read1_fa_path = f'/home/dev_8T/xlxiang/starr-seq/20250124_YaoTangShangHaiShengWuKeJiYouXianGongSi-yaoqi-1_1/00.mergeRawFq/{sample}/{sample}_1_trim_paired.fa'
read2_fa_path = f'/home/dev_8T/xlxiang/starr-seq/20250124_YaoTangShangHaiShengWuKeJiYouXianGongSi-yaoqi-1_1/00.mergeRawFq/{sample}/{sample}_2_trim_paired.fa'
#%%
def read_fa(path):
    """Parses a FASTA file and returns a DataFrame with 'Name' and 'Sequence'."""
    names = []
    sequences = []
    with open(path, 'r') as f:
        name = None
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    names.append(name)
                    sequences.append(seq)
                name = line[1:]
                seq = ""
            else:
                seq += line
        if name:
            names.append(name)
            sequences.append(seq)

    return pd.DataFrame({"Name": names, "Sequence": sequences})

def rever_seq(seqs):   
    """Returns the Reverse Complement of a list of DNA sequences."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'} 
    rev_comp_list = []
    for s in seqs:
        rev_comp = "".join([complement.get(base, base) for base in s[::-1]])
        rev_comp_list.append(rev_comp)
    return rev_comp_list

#%%
# Process Read 1 (R1)
df_1 = read_fa(read1_fa_path)

# Extract the first 10bp as an index/prefix for QC
df_1['index'] = [i[:10] for i in df_1['Sequence']]
print(f"Total R1 reads: {df_1.shape[0]}")
print(f"Ratio of most frequent R1 index: {np.max(list(Counter(df_1['index']).values()))/df_1.shape[0]:.4f}")

# Clean R1 names for merging
df_1['Read_n'] = [i[:-2] for i in df_1['Name']]

#%%
# Process Read 2 (R2)
df_2 = read_fa(read2_fa_path)

# Extract the first 16bp as an index for QC
df_2['index'] = [i[:16] for i in df_2['Sequence']]
print(f"Total R2 reads: {df_2.shape[0]}")
print(f"Ratio of most frequent R2 index: {np.max(list(Counter(df_2['index']).values()))/df_2.shape[0]:.4f}")

# Clean R2 names for merging
df_2['Read_n'] = [i[:-2] for i in df_2['Name']]

# Barcode Extraction (Initial 12bp check for classification)
# Using a fixed offset of 106bp based on library design
df_2['bar'] = [i[106:106+12].upper() for i in df_2['Sequence']]
df_2 = df_2[(df_2['bar'].str.len() == 12) & (df_2['bar'].str.match(r'^[ATCG]+$'))]

# Define the control (35s) barcode
mini_bar = 'TAGGGCCGGTAG'
df_2['type_2'] = np.where(df_2['bar'] == mini_bar, '35s', 'syn')

# Expand barcode extraction to 15bp for high-resolution analysis
df_2['bar'] = [i[106:106+15].upper() for i in df_2['Sequence']]
df_2 = df_2[(df_2['bar'].str.len() == 15) & (df_2['bar'].str.match(r'^[ATCG]+$'))]

#%%
# Merge R1 and R2
df_merge = df_1.merge(df_2, on='Read_n')
print(f"Total valid paired reads: {df_merge.shape[0]}")

# Load Bowtie2 alignment results (mapping reads to gene names)
bowtie2_result = pd.read_csv(f'{result_path}/pro.tsv', sep='\t')
# Filter alignments by start position (ensure mapping is at the beginning of the promoter)
bowtie2_result = bowtie2_result[bowtie2_result.start <= 80]
bowtie2_result = bowtie2_result.loc[:, ['read', 'gene']].drop_duplicates()

# Final merge: Pair sequencing data with alignment metadata
df_merge_last = df_merge.merge(bowtie2_result, left_on='Read_n', right_on='read')

# Subgroup 1: Synthetic library (not 35s)
syn_df = df_merge_last[(df_merge_last['type_2'] == 'syn') & (df_merge_last['gene'] != '35s')]
syn_df = syn_df.loc[:, ['bar', 'gene']]

# Subgroup 2: Clear 35s control (type matches alignment)
mini_df = df_merge_last[(df_merge_last['type_2'] != 'syn') & (df_merge_last['gene'] == '35s')]
mini_df = mini_df.loc[:, ['bar', 'gene']]
mini_df['bar'] = mini_bar  # Standardize barcode to 12bp for control

# Subgroup 3: Fuzzy 35s recovery
# Check reads that aligned to 35s but failed the exact barcode match due to sequencing errors
mini_df_1 = df_merge_last[(df_merge_last['type_2'] == 'syn') & (df_merge_last['gene'] == '35s')].copy()
mini_df_1['bar_prefix'] = [i[:-3] for i in mini_df_1['bar']]
# Calculate Levenshtein distance to recover mutated control barcodes (distance < 2)
mini_df_1['leven_min'] = [Levenshtein.distance(i, mini_bar) for i in mini_df_1['bar_prefix']]
mini_df_1 = mini_df_1[mini_df_1.leven_min < 2]
mini_df_1['bar'] = mini_bar
mini_df_1 = mini_df_1.loc[:, ['bar', 'gene']]

# Combine all valid groups
repeat_last = pd.concat([syn_df, mini_df, mini_df_1])

print(f"Alignment success rate (matched reads / total valid): {repeat_last.shape[0]/df_merge.shape[0]:.4f}")

# Export final mapping file
repeat_last.to_csv(f'{result_path}/bar_pro.txt', index=False)
print('Process completed successfully!')