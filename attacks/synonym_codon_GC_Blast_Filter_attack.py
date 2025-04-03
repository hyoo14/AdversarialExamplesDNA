# from google.colab import drive
# drive.mount("/content/drive")


# !pip install selenium
# !apt-get update

# # (최초 1회)
# !apt install chromium-chromedriver
# !cp /usr/lib/chromium-browser/chromedriver '/content/drive/MyDrive/Colab Notebooks' #
# !pip install chromedriver-autoinstaller


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import sys
from selenium.webdriver.common.keys import Keys
import urllib.request
import os
from urllib.request import urlretrieve

import time
import pandas as pd
import chromedriver_autoinstaller  # setup chrome options


chrome_path = "/content/drive/MyDrive/Colab Notebooks/chromedriver"
sys.path.insert(0,chrome_path)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # ensure GUI is off
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')  # set path to chromedriver as per your configuration
chrome_options.add_argument('lang=ko_KR') # 한국어

chromedriver_autoinstaller.install()  # set the target URL
from selenium.webdriver.chrome.service import Service

service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)

# BLAST 사이트 접속
url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
driver.get(url)

# 페이지 타이틀 가져오기
title = driver.title
print("페이지 타이틀:", title)
# 드라이버 종료
driver.quit()


# Input Sequence
input_seq = "ATGAATCCCTATCTACAGTTAGGTCACAAAGAATTTTCGTTAGAAAAAAACTAAGAACCCCCTCATTTAGTCCTTGCTGCCTTCAGCGGAGTCGAGGTTTAGTTGCAGCCAGAATCAGCCAAACAATGCTAACGGCTCGTCAAAGCCTTAAAGCTCGTAAACGAGATCTGCTTGTCAGATGGCTACCGAACCGAAAAGCAGCAACGGTATTTATGGGAATATTCCATGATAGAAAATGGGCTAGCCTCTATGAAACAATTTGTGGCATTGCCCGGTCGCAGTCAACATCAGTTAGGCTTAGCCCTCAATTTTGGTTTAAAGGGCAGCCAGGTAGATTTTATCTGCCCAGTATTTCGGGACAGCGCAGCCGCTGATTTATTTACCCAGGAAATGCTTAACTATGGGTTTATTTTAGGCTATTCCGCAGACAAACAGGAGATTTCAGGGATTGGCTGTGAACCTTGTCATTTCCGGTATGTCTGGCTGCCTCATAGGCAAATCATCGCCAGTCAACAGTGGACCTAGGAAGAAGCCCATCAATACCGTCATCAAACTGCGGGGCAGTTCGCATGA"

# BLAST URL with query
blast_url = f"https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROGRAM=blastn&LINK_LOC=gquery&QUERY={input_seq}"


service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)

# BLAST 페이지 접속
driver.get(blast_url)

# BLAST 버튼 클릭 (XPath 방식)
try:
    blast_button = WebDriverWait(driver, 10).until(

        EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div/div[2]/form/div[6]/div/div[1]/div[1]/input"))
        # EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[2]/form/div[6]/div/div[1]/div[1]/input"))
    )
    blast_button.click()
    print("BLAST 버튼 클릭 성공")
except Exception as e:
    print("BLAST 버튼 클릭 실패:", e)
    driver.quit()
    exit()

# 검색 결과 페이지 대기 (최대 30초)
try:
    result_element = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/main/section[2]/ul/li[1]/div/div[3]/form/div[3]/div/table/tbody/tr[1]/td[3]/span/a"))
    )

    # title 속성 추출
    result_title = result_element.get_attribute("title")
    print("첫 번째 검색 결과:", result_title)

except Exception:
    print("No result")

# 드라이버 종료
driver.quit()


def species_blast(input_seq):
  # Input Sequence
  #input_seq = "ATGAATCCCTATCTACAGTTAGGTCACAAAGAATTTTCGTTAGAAAAAAACTAAGAACCCCCTCATTTAGTCCTTGCTGCCTTCAGCGGAGTCGAGGTTTAGTTGCAGCCAGAATCAGCCAAACAATGCTAACGGCTCGTCAAAGCCTTAAAGCTCGTAAACGAGATCTGCTTGTCAGATGGCTACCGAACCGAAAAGCAGCAACGGTATTTATGGGAATATTCCATGATAGAAAATGGGCTAGCCTCTATGAAACAATTTGTGGCATTGCCCGGTCGCAGTCAACATCAGTTAGGCTTAGCCCTCAATTTTGGTTTAAAGGGCAGCCAGGTAGATTTTATCTGCCCAGTATTTCGGGACAGCGCAGCCGCTGATTTATTTACCCAGGAAATGCTTAACTATGGGTTTATTTTAGGCTATTCCGCAGACAAACAGGAGATTTCAGGGATTGGCTGTGAACCTTGTCATTTCCGGTATGTCTGGCTGCCTCATAGGCAAATCATCGCCAGTCAACAGTGGACCTAGGAAGAAGCCCATCAATACCGTCATCAAACTGCGGGGCAGTTCGCATGA"

  # BLAST URL with query

  blast_url = f"https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROGRAM=blastn&LINK_LOC=gquery&QUERY={input_seq}"


  service = Service()
  driver = webdriver.Chrome(service=service, options=chrome_options)

  # BLAST 페이지 접속
  driver.get(blast_url)

  # BLAST 버튼 클릭 (XPath 방식)
  try:
      blast_button = WebDriverWait(driver, 10).until(
          EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div/div[2]/form/div[6]/div/div[1]/div[1]/input"))
          #EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[2]/form/div[6]/div/div[1]/div[1]/input"))
      )
      blast_button.click()
      print("BLAST 버튼 클릭 성공")
  except Exception as e:
      print("BLAST 버튼 클릭 실패:", e)
      driver.quit()
      exit()

  # 검색 결과 페이지 대기 (최대 30초)
  try:
      result_element = WebDriverWait(driver, 30).until(
          EC.presence_of_element_located((By.XPATH, "/html/body/main/section[2]/ul/li[1]/div/div[3]/form/div[3]/div/table/tbody/tr[1]/td[3]/span/a"))
      )

      # title 속성 추출
      result_title = result_element.get_attribute("title")
      print(result_title)
      result = result_title

  except Exception:
      print("No result")
      result = "No result"

  # 드라이버 종료
  driver.quit()
  return result

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


def calculate_gc_content(seq):
    g = seq.count('G')
    c = seq.count('C')
    return (g + c) / len(seq)

# def synonymous_filter_codon_attack(sequences, mutation_rate=0.1, iteration=1):
#     mutated_sequences = sequences.copy()
#     for _ in range(iteration):

#         mutated_sequences = mutated_sequences.apply(
#             lambda seq: codon_synonymous_mutation(seq, mutation_rate)
#         )
#     return mutated_sequences
def two_filter_gen(seq, mutation_rate=0.1, max_iter=10):
  original_gc = calculate_gc_content(seq)
  original_taxa = species_blast(seq)

  flag=0

  for i in range(max_iter):
    new_seq = codon_synonymous_mutation(seq, mutation_rate)
    new_gc, new_taxa = calculate_gc_content(new_seq), species_blast(new_seq)

    if new_taxa == original_taxa and abs(new_gc - original_gc) < 0.05:
      return new_seq

  return ""




def synonymous_filter_codon_attack(sequences, mutation_rate=0.1, iteration=1):
    mutated_sequences = sequences.copy()

    for it in range(iteration):
        print(f"\n[Iteration {it+1}/{iteration}]")
        new_sequences = []
        for idx, seq in enumerate(mutated_sequences):
            print(f"Processing sequence {idx+1}/{len(mutated_sequences)}")

            new_seq = two_filter_gen(seq, mutation_rate)

            new_sequences.append(new_seq)

            if idx % 100 == 0:
                temp_df = pd.DataFrame({'DNA Sequence': new_sequences})
                temp_df.to_csv(f"/content/drive/MyDrive/RDL/prj/data/TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv", index=False)
                print(f"/content/drive/MyDrive/RDL/prj/data/TEMP_test_AMR_atk_synonym_filter_codon_iter1_rate0_{int(mutation_rate*10)}.csv file saved(temp)")

        mutated_sequences = pd.Series(new_sequences)

    return mutated_sequences


# for i in [5]:#range(2, 6):
#   test_dir = "/content/drive/MyDrive/applicationsML/nlp_project/datasets/df9class_CARD_MEGARes_test_dc.csv"
#   test_df = pd.read_csv(f"{test_dir}")
#   test_df['DNA Sequence'] = test_df['DNA Sequence'].str.upper()

#   iternum = 1
#   mutation_rate= 0.1 * i

#   test_df['DNA Sequence'] = synonymous_filter_codon_attack(test_df['DNA Sequence'], mutation_rate, iternum)

#   test_df.to_csv(f"/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_synonym_filter_codon_iter1_rate0_{i}.csv")
#   print(f"/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_synonym_filter_codon_iter1_rate0_{i}.csv file saved with rate {mutation_rate}!")

