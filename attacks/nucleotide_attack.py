import random
import pandas as pd


# Define mutation functions
def nucleotide_mutation(sequence, mutation_rate=0.1):
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            sequence[i] = random.choice('ATCG')
    return ''.join(sequence)


def nucleotide_attack(sequences, mutation_rate=0.1, iteration=1):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: nucleotide_mutation(seq, mutation_rate)
        )
  return mutated_sequences



# test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
# test_df = pd.read_csv(f"{test_dir}")
# test_df['sequence'] = test_df['sequence'].str.upper()

# iternum = 1
# mutation_rate=0.1

# test_df['sequence'] = nucleotide_attack(test_df['sequence'], mutation_rate, iternum)
# test_df.to_csv("/content/drive/MyDrive/RDL/prj/data/test_atk_nucl_iter1_rate0_1.csv")
