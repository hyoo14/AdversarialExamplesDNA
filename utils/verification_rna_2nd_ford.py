# from google.colab import drive
# drive.mount('/content/drive')

# RNA 2nd ford
# !apt-get install -y vienna-rna

# 데이터 불러오기
df_original = pd.read_csv("/content/drive/MyDrive/applicationsML/nlp_project/datasets/df9class_CARD_MEGARes_test_dc.csv")
df_attack = pd.read_csv("/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_synonym_codon_iter1_rate0_1.csv")




import pandas as pd
import subprocess
from tqdm import tqdm  # 진행률 바

# RNAfold로 MFE 계산 함수
def get_mfe(seq: str) -> float:
    rna_seq = seq.upper().replace('T', 'U')
    result = subprocess.run(
        ['RNAfold'],
        input=rna_seq,
        capture_output=True,
        text=True
    )
    lines = result.stdout.strip().split('\n')
    mfe_line = lines[-1]
    mfe_value = float(mfe_line.split('(')[-1].strip(')'))
    return mfe_value



# 전처리: 시퀀스를 대문자로 통일하고 길이 맞추기
df_original = df_original.copy()
df_original["DNA Sequence"] = df_original["DNA Sequence"].str.upper()
df_original = df_original.iloc[:len(df_attack)].reset_index(drop=True)
df_attack = df_attack.iloc[:len(df_original)].reset_index(drop=True)

# ΔMFE 계산
delta_mfes = []
for ori_seq, atk_seq in tqdm(zip(df_original["DNA Sequence"], df_attack["DNA Sequence"]), total=len(df_attack)):
    try:
        mfe_ori = get_mfe(ori_seq)
        mfe_atk = get_mfe(atk_seq)
        delta_mfe = mfe_atk - mfe_ori
        delta_mfes.append(delta_mfe)
    except Exception as e:
        print(f"Error processing sequence: {e}")
        delta_mfes.append(None)

# 평균 ΔMFE 계산
valid_deltas = [d for d in delta_mfes if d is not None]
mean_delta = sum(valid_deltas) / len(valid_deltas)

print(f"Mean ΔMFE: {mean_delta:.4f}")
