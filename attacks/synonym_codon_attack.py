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

def codon_synonymous_mutation(sequence, mutation_rate=0.1):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    for i in range(len(codons)):
        codon = codons[i]
        if len(codon) == 3 and codon in codon_table:
            if random.random() < mutation_rate:
                aa = codon_table[codon]
                synonyms = aa_to_codons[aa]
                if len(synonyms) > 1:
                    # 현재 코돈 제외하고 무작위 선택
                    alternatives = [c for c in synonyms if c != codon]
                    codons[i] = random.choice(alternatives)
    return ''.join(codons)

def synonymous_codon_attack(sequences, mutation_rate=0.1, iteration=1):
    mutated_sequences = sequences.copy()
    for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_synonymous_mutation(seq, mutation_rate)
        )
    return mutated_sequences



#test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
# test_dir = "/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv"
# test_df = pd.read_csv(f"{test_dir}")
# test_df['DNA Sequence'] = test_df['DNA Sequence'].str.upper()

# iternum = 1
# mutation_rate=0.1

# test_df['DNA Sequence'] = synonymous_codon_attack(test_df['DNA Sequence'], mutation_rate, iternum)

