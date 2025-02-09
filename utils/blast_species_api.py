

# !pip install selenium
# !apt-get update

# # (최초 1회)
# !apt install chromium-chromedriver
# !cp /usr/lib/chromium-browser/chromedriver '/content/drive/MyDrive/Colab Notebooks' #
# !pip install chromedriver-autoinstaller

# !python --version

import selenium
print(selenium.__version__)




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
          EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[2]/form/div[6]/div/div[1]/div[1]/input"))
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


csv_path = "/content/drive/MyDrive/RDL/prj/data/test_AMR_atk_NT_bt_iter300_rate0.1.csv"
output_csv_path = "/content/drive/MyDrive/RDL/prj/data_blast/blast_results_test_AMR_atk_NT_bt_iter300_rate0.1.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# "DNA Sequence" 컬럼에 대해 BLAST 실행하여 결과 추가
blast_results = []

idx = 0
for seq in df["DNA Sequence"]:
    idx +=1
    # if idx < 59:
    #   continue

    result = species_blast(seq)
    blast_results.append(result)
    print(f"{idx}... -> {result}")  # 시퀀스 일부와 결과 출력 (실시간)


# 결과를 DataFrame에 추가
df["BLAST Result"] = blast_results

# 결과 저장
df.to_csv(output_csv_path, index=False)

print("BLAST 검색 결과 저장 완료:", output_csv_path)








##### version 2 for cluster #####
## need to download chrome drive and make aligned with specific directories
# !mkdir -p ~/chrome
# !wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O chrome.deb
# !ar x chrome.deb
# !tar -xf data.tar.xz
# !~/chrome/opt/google/chrome/google-chrome --version
# !/home/x-hyoo2/chrome/opt/google/chrome/google-chrome --version
# !chmod +x /home/x-hyoo2/chrome/chromedriver



# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service

# options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # GUI 없는 환경에서 실행
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")

# # Chrome 바이너리 경로 지정
# options.binary_location = "/home/x-hyoo2/chrome/opt/google/chrome/google-chrome"

# # Chromedriver 경로 직접 지정
# chromedriver_path = "/home/x-hyoo2/chrome/chromedriver/chromedriver"  # Chromedriver가 위치한 경로
# service = Service(chromedriver_path)

# # WebDriver 실행
# driver = webdriver.Chrome(service=service, options=options)

# driver.get("https://www.google.com")
# print(" Selenium 실행 성공:", driver.title)

# driver.quit()


# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.by import By

# #input_seq = "ATGAATCCCTATCTACAGTTAGGTCACAAAGAATTTTCGTTAGAAAAAAACTAAGAACCCCCTCATTTAGTCCTTGCTGCCTTCAGCGGAGTCGAGGTTTAGTTGCAGCCAGAATCAGCCAAACAATGCTAACGGCTCGTCAAAGCCTTAAAGCTCGTAAACGAGATCTGCTTGTCAGATGGCTACCGAACCGAAAAGCAGCAACGGTATTTATGGGAATATTCCATGATAGAAAATGGGCTAGCCTCTATGAAACAATTTGTGGCATTGCCCGGTCGCAGTCAACATCAGTTAGGCTTAGCCCTCAATTTTGGTTTAAAGGGCAGCCAGGTAGATTTTATCTGCCCAGTATTTCGGGACAGCGCAGCCGCTGATTTATTTACCCAGGAAATGCTTAACTATGGGTTTATTTTAGGCTATTCCGCAGACAAACAGGAGATTTCAGGGATTGGCTGTGAACCTTGTCATTTCCGGTATGTCTGGCTGCCTCATAGGCAAATCATCGCCAGTCAACAGTGGACCTAGGAAGAAGCCCATCAATACCGTCATCAAACTGCGGGGCAGTTCGCATGA"
# input_seq = "GGGAATCCCTATTAGTGCTTAGTTGCCAAAGTTGTACCGTTAGAAAAAAAGCATGAACAGAACCAATTAGATCTTGCTGCCTTCAGCGAAGTGGAGCTACATTTGCAGTCATTGGCAGGTAGTCAATGGGAAAACTGCAAAAAAGAGTTAAAGCCAGACAACGAGATCCAATTGCGGGATGACTACCGAACCGAAAAGTGGAATCGGTATTACTGGGAATATTCCTTGAAAGAAAATATTCTAGAATATACGTCGCAAAGGCCCTCTTCGCCCGGTTGCGCTGAACATCAGTTAGGCTTACCGATCGATGTTGGTTTAAAGCTGAGCCAGGATGGGAAACGGTGCATGCTATTTAATGACAGCGCAGCCTTTGATTTATTTACCATAGTGCTGATGAACTATTGAACCATTTTACTCTATCCCTGCGCAAAACAGGAGATTCTAGGGATTGGCCGAGAACGCTGGCATTTCCCCTCTGTCGGGTCATCGCATAGTATGATCAGTGCCAGTCAACAGATGACCTTGGAAGATTACCATCAACGGCTTCGGTTCACTGCGAGGAAATTCGCATGA"

# blast_url = f"https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROGRAM=blastn&LINK_LOC=gquery&QUERY={input_seq}"


# service = Service()
# driver = webdriver.Chrome(service=service, options=options)

# # BLAST 페이지 접속
# driver.get(blast_url)

# # BLAST 버튼 클릭 (XPath 방식)
# try:
#     blast_button = WebDriverWait(driver, 10).until(
#         EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[2]/form/div[6]/div/div[1]/div[1]/input"))
#     )
#     blast_button.click()
#     print("BLAST 버튼 클릭 성공")
# except Exception as e:
#     print("BLAST 버튼 클릭 실패:", e)
#     driver.quit()
#     exit()

# # 검색 결과 페이지 대기 (최대 30초)
# try:
#     result_element = WebDriverWait(driver, 30).until(
#         EC.presence_of_element_located((By.XPATH, "/html/body/main/section[2]/ul/li[1]/div/div[3]/form/div[3]/div/table/tbody/tr[1]/td[3]/span/a"))
#     )

#     # title 속성 추출
#     result_title = result_element.get_attribute("title")
#     print(result_title)
#     result = result_title

# except Exception:
#     print("No result")
#     result = "No result"

# # 드라이버 종료
# driver.quit()


