import pandas as pd
import numpy as np
from collections import Counter

# File paths and configuration
sample = 'SPO-C1R1-0120'
result_path = '/home/xlxiang/last_TargetGAN/figplot_code/chapter_5_starr_seq/script/repeat_1'
read1_fa_path = f'/home/dev_8T/xlxiang/starr-seq/20250124_YaoTangShangHaiShengWuKeJiYouXianGongSi-yaoqi-1_1/00.mergeRawFq/{sample}/{sample}_1_trim_paired.fa'
read2_fa_path = f'/home/dev_8T/xlxiang/starr-seq/20250124_YaoTangShangHaiShengWuKeJiYouXianGongSi-yaoqi-1_1/00.mergeRawFq/{sample}/{sample}_2_trim_paired.fa'

def read_fa(path):
    """Parses a FASTA file into a pandas DataFrame with 'Name' and 'Sequence' columns."""
    names = []
    sequences = []
    with open(path, 'r') as f:
        name = None
        seq = None
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
    """Generates the Reverse Complement of a list of DNA sequences."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'} 
    revers_seq = []
    for i in seqs:
        # Step 1: Reverse the string; Step 2: Map to complement bases
        reversed_sequence = i[::-1]
        complement_sequence = ''.join([complement.get(base, base) for base in reversed_sequence])
        revers_seq.append(complement_sequence)
    return revers_seq

# Process Read 1
df_1 = read_fa(read1_fa_path)
print(f"Total reads in R1: {df_1.shape[0]}")

# Strip suffix from read names to enable merging (e.g., removing /1 or /2)
df_1['Read_n'] = [i[:-2] for i in df_1['Name']]

# Extract potential barcode (12bp) for type classification
df_1['bar_1'] = [i[64:64+12].upper() for i in df_1['Sequence']]
# Filter: Ensure length is 12 and contains only valid DNA bases
df_1 = df_1[(df_1['bar_1'].str.len() == 12) & (df_1['bar_1'].str.match(r'^[ATCG]+$'))]

# Classification: '35s' if it matches the fixed control sequence, otherwise 'syn' (synthetic)
df_1['type_1'] = np.where(df_1['bar_1'] == 'CTACCGGCCCTA', '35s', 'syn')

# Refine barcode extraction to 15bp for downstream analysis
df_1['bar_1'] = [i[64:64+15].upper() for i in df_1['Sequence']]
df_1 = df_1[(df_1['bar_1'].str.len() == 15) & (df_1['bar_1'].str.match(r'^[ATCG]+$'))]

counts_1 = Counter(df_1['type_1'])
print(f"R1 Classification: {counts_1}")
print(f"35s Ratio in R1: {counts_1['35s'] / sum(counts_1.values()):.4f}")

# Process Read 2
df_2 = read_fa(read2_fa_path)
df_2['Read_n'] = [i[:-2] for i in df_2['Name']]

# Extract potential barcode from R2 (offset based on library structure)
df_2['bar_2'] = [i[82+21:82+21+12].upper() for i in df_2['Sequence']]
df_2 = df_2[(df_2['bar_2'].str.len() == 12) & (df_2['bar_2'].str.match(r'^[ATCG]+$'))]

# Classification for R2: note 'TAGGGCCGGTAG' is likely the RC of 'CTACCGGCCCTA'
df_2['type_2'] = np.where(df_2['bar_2'] == 'TAGGGCCGGTAG', '35s', 'syn')

# Refine barcode extraction to 15bp for R2
df_2['bar_2'] = [i[82+21:82+21+15].upper() for i in df_2['Sequence']]
df_2 = df_2[(df_2['bar_2'].str.len() == 15) & (df_2['bar_2'].str.match(r'^[ATCG]+$'))]

counts_2 = Counter(df_2['type_2'])
print(f"R2 Classification: {counts_2}")
print(f"35s Ratio in R2: {counts_2['35s'] / sum(counts_2.values()):.4f}")

# Merge R1 and R2 dataframes on unique Read ID
df_merged = df_1.merge(df_2, on='Read_n', how='inner').set_index('Read_n')

# Keep only reads where R1 and R2 agree on the classification type
df_merged = df_merged[df_merged['type_1'] == df_merged['type_2']]

# Handle 'syn' library: Reverse complement R2 barcode and verify it matches R1
syn_df = df_merged[df_merged['type_1'] == 'syn'].copy()
syn_df['rever_bar'] = rever_seq(list(syn_df['bar_2']))
syn_df = syn_df[syn_df['rever_bar'] == syn_df['bar_1']]

# Handle '35s' control: Standardize the barcode representation
df_35s = df_merged[df_merged['type_1'] == '35s'].copy()
df_35s['rever_bar'] = 'CTACCGGCCCTA'

# Final data aggregation
last_syn = syn_df.loc[:, ['rever_bar']]
last_35s = df_35s.loc[:, ['rever_bar']]
result = pd.concat([last_syn, last_35s])

# Export results to text file
result.to_csv(f'{result_path}/cdna.txt')
print('Process completed successfully!')