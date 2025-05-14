# from google.colab import drive
# drive.mount("/content/drive")


import random
import pandas as pd

# 아미노산 ↔ 코돈 매핑
codon_table = {
    'TTT': 'F', 'TTC': 'F',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y',
    'CAT': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C',
    'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}

# 아미노산 → 가능한 코돈들
aa_to_codons = {}
for codon, aa in codon_table.items():
    if aa not in aa_to_codons:
        aa_to_codons[aa] = []
    aa_to_codons[aa].append(codon)

def gc_content(seq):
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / len(seq)

def gc_difference(seq1, seq2):
    return abs(gc_content(seq1) - gc_content(seq2))

def codon_synonymous_gc_guided(sequence, mutation_rate=0.1, lambda_gc=1.0):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    original_gc = gc_content(sequence)
    for i in range(len(codons)):
        codon = codons[i]
        if len(codon) == 3 and codon in codon_table:
            if random.random() < mutation_rate:
                aa = codon_table[codon]
                synonyms = aa_to_codons[aa]
                if len(synonyms) > 1:
                    best_score = float('-inf')
                    best_codon = codon
                    for alt in synonyms:
                        if alt == codon:
                            continue
                        temp_codons = codons.copy()
                        temp_codons[i] = alt
                        mutated_seq = ''.join(temp_codons)
                        score = -lambda_gc * gc_difference(sequence, mutated_seq)
                        if score > best_score:
                            best_score = score
                            best_codon = alt
                    codons[i] = best_codon
    return ''.join(codons)

def synonymous_codon_attack_gc(sequences, mutation_rate=0.1, lambda_gc=1.0, iteration=1):
    mutated_sequences = sequences.copy()
    for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_synonymous_gc_guided(seq, mutation_rate, lambda_gc)
        )
    return mutated_sequences



#test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
# test_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv"
# test_df = pd.read_csv(f"{test_dir}")
# test_df['DNA Sequence'] = test_df['DNA Sequence'].str.upper()

# iternum = 1
# mutation_rate=0.1

# test_df['DNA Sequence'] = synonymous_codon_attack_gc(test_df['DNA Sequence'], mutation_rate, iternum)

# test_df.to_csv("/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_synonym_codon_GC_guide_iter1_rate0_1.csv")

for i in range(2, 6):
  test_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv"
  test_df = pd.read_csv(f"{test_dir}")
  test_df['DNA Sequence'] = test_df['DNA Sequence'].str.upper()

  iternum = 1
  mutation_rate= 0.1 * i

  test_df['DNA Sequence'] = synonymous_codon_attack_gc(test_df['DNA Sequence'], mutation_rate, iternum)

  test_df.to_csv(f"/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_synonym_codon_GC_guide_iter1_rate0_{i}.csv")
  print(f"/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_synonym_codon_GC_guide_iter1_rate0_{i}.csv file saved with rate {mutation_rate}!")

