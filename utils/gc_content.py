def calculate_gc_content(sequence):
    """주어진 DNA 서열의 GC 함량을 계산하는 함수"""
    gc_count = sequence.count('G') + sequence.count('C')
    total_length = len(sequence)
    gc_content = (gc_count / total_length) * 100 if total_length > 0 else 0
    return gc_content

def predict_promoter_potential(sequence, gc_threshold=50):
    """
    주어진 서열의 GC 함량을 기반으로 프로모터 가능성을 예측하는 함수
    - gc_threshold: 프로모터 가능성이 높다고 판단하는 GC 함량 기준
    """
    gc_content = calculate_gc_content(sequence)

    if gc_content >= gc_threshold:
        prediction = "높은 가능성 (GC-rich 영역)"
    else:
        prediction = "낮은 가능성 (AT-rich 영역)"

    return gc_content, prediction

# 예제 서열
dna_sequence = "AAAAAAACCCCCCCCCCCCCCCCCCCTTTTTTTTTTTTGGGGGG"

# GC 함량 계산 및 프로모터 가능성 예측
gc_content, prediction = predict_promoter_potential(dna_sequence)

# 결과 출력
gc_content, prediction

import pandas as pd
# CSV 파일 불러오기
file_path = "/content/drive/MyDrive/RDL/prj/data/test.csv"
df = pd.read_csv(file_path)

# GC 함량 계산 후 새로운 컬럼 추가
df["GC_Content"] = df["sequence"].apply(calculate_gc_content)


file_path2 = "/content/drive/MyDrive/RDL/prj/data/test_PD_atk_nucl_iter1_rate0_1.csv"
df2 = pd.read_csv(file_path2)

# GC 함량 계산 후 새로운 컬럼 추가
df2["GC_Content"] = df2["sequence"].apply(calculate_gc_content)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# df2가 존재한다고 가정하고 불러오기 (같은 방식으로 GC_Content 계산됨)
# df["GC_Content"], df2["GC_Content"]이 존재한다고 가정

# 1평균 GC 함량 비교
mean_diff = abs(df["GC_Content"].mean() - df2["GC_Content"].mean())

# 코사인 유사도 계산 (1D 배열을 2D로 reshape 필요)
gc_content_df = np.array(df["GC_Content"]).reshape(1, -1)
gc_content_df2 = np.array(df2["GC_Content"]).reshape(1, -1)
cosine_sim = cosine_similarity(gc_content_df, gc_content_df2)[0][0]

# 피어슨 상관계수 계산
pearson_corr = np.corrcoef(df["GC_Content"], df2["GC_Content"])[0, 1]

# 결과 출력
similarity_results = {
    "Mean Difference": mean_diff,
    "Cosine Similarity": cosine_sim,
    "Pearson Correlation": pearson_corr
}

similarity_results
