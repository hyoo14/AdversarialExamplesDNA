import random
import pandas as pd


# Define mutation functions

# def nucleotide_mutation(sequence, mutation_rate=0.1):
#     sequence = list(sequence)
#     for i in range(len(sequence)):
#         if random.random() < mutation_rate:
#             sequence[i] = random.choice('ATCG')
#     return ''.join(sequence)

def codon_mutatation(sequence, mutation_rate=0.1):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    for i in range(len(codons)):
        if random.random() < mutation_rate:
            codons[i] = ''.join(random.choices('ATCG', k=3))
    return ''.join(codons)


def nucleotide_attack(sequences, mutation_rate=0.1, iteration=1):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_mutatation(seq, mutation_rate)
        )
  return mutated_sequences



# test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
# test_df = pd.read_csv(f"{test_dir}")
# test_df['sequence'] = test_df['sequence'].str.upper()

# iternum = 1
# mutation_rate=0.1

# test_df['sequence'] = nucleotide_attack(test_df['sequence'], mutation_rate, iternum)

# test_df.to_csv("/content/drive/MyDrive/RDL/prj/data/test_atk_codon_iter1_rate0_1.csv")
