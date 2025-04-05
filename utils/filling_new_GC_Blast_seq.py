# below part is in the attcks directory
## use the file name "synonym_codon_GC_Blast_Filter_attack.py"


# making sequence for filling empty one

import pandas as pd

mutation_rate = 0.3
now_df = pd.read_csv(f"/content/drive/MyDrive/RDL/prj/data/PART2_FILLED_TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv")
now_df["len"] = now_df["DNA Sequence"].apply(lambda x: len(str(x)))
idx_list = now_df[now_df["len"] <= 4].index.tolist()

print(len(idx_list))
print(idx_list)



# 인덱스 기준으로 "DNA Sequence" 컬럼만 추출
df_orig = pd.read_csv("/content/drive/MyDrive/playground_test_anything/bio_AMR_ARG/_NT_test/df9class_CARD_MEGARes_test_dc.csv")
df_orig['DNA Sequence'] = df_orig['DNA Sequence'].str.upper()

dna_sequences = df_orig.loc[idx_list, "DNA Sequence"]

# 결과 출력
print(dna_sequences)

new_sequences = []
cnt = 0
for seq in dna_sequences:
  cnt+=1
  # if cnt < 54:
  #   continue
  print(f"***************** now cnt:  ( {cnt} )")
  new_seq = two_filter_gen(seq, mutation_rate, max_iter=160)
  if len(new_seq) > 5: print(f"*****************************************************new sequence generate success! ")
  new_sequences.append(new_seq)

temp_df = pd.DataFrame({'DNA Sequence': new_sequences})
temp_df.to_csv(f"/content/drive/MyDrive/RDL/prj/data/FILL_PART2_TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv", index=False)


# filling process

import pandas as pd

mutation_rate = 0.3
now_df = pd.read_csv(f"/content/drive/MyDrive/RDL/prj/data/PART2_FILLED_TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv")
now_df["len"] = now_df["DNA Sequence"].apply(lambda x: len(str(x)))
idx_list = now_df[now_df["len"] <= 4].index.tolist()

len(idx_list)

new_df = pd.read_csv(f"/content/drive/MyDrive/RDL/prj/data/FILL_PART1_TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv")

for i, idx in enumerate(idx_list):
    print(f" i: {i},  idx: {idx}")
    now_df.loc[idx, "DNA Sequence"] = new_df.iloc[i]["DNA Sequence"]


now_df.to_csv(f"/content/drive/MyDrive/RDL/prj/data/PART2_FILLED_TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv", index=False)
