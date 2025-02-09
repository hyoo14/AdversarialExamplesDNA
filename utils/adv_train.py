import os
import sys
import subprocess

# Hugging Face 모델 저장 경로 설정
os.environ["HF_HOME"] = "huggingface_models"

# pip 패키지 저장 경로 설정 (이 경로만 사용하게 강제)
pip_target = "libraries"
os.environ["PYTHONUSERBASE"] = pip_target
os.environ["PYTHONPATH"] = f"{pip_target}/lib/python3.10/site-packages:" + os.environ.get("PYTHONPATH", "")

# sys.path에 추가해서 즉시 반영
sys.path.insert(0, f"{pip_target}/lib/python3.10/site-packages")



# GROVER and NT and DNABERT2
# Imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, BertConfig
import torch
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import matplotlib.pyplot as plt
from datetime import datetime
import random
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import HfApi, login, hf_hub_download, create_repo
from safetensors.torch import save_file, load_file



# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 및 Tokenizer 설정



# Constants




## PD
# task_name = "PD"
# test_file_path = test_dir = "/content/drive/MyDrive/RDL/prj/data/test.csv"
# train_dir = "/content/drive/MyDrive/RDL/prj/data/train.csv"
# valid_dir = "/content/drive/MyDrive/RDL/prj/data/dev.csv"

# test_label_name = "label"

# train_df = pd.read_csv(train_dir)
# label_to_id = {label: i for i, label in enumerate(train_df[test_label_name].unique())}# Create label-to-id mapping


#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "GROVER", "hyoo14/GROVER_PD", 512, 32, "sequence" #GROVER               OK!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "DNABERT2", "hyoo14/DNABERT2_PD", 510, 32, "sequence" #DNABERT2          OK!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "NT", "hyoo14/NucletideTransformer_PD", 1000, 32, "sequence" #NT       worker

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "GROVER", "hyoo14/GROVER_PD", 512, 32, "sequence" #GROVER             worker
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "DNABERT2", "hyoo14/DNABERT2_PD", 510, 32, "sequence" #DNABERT2        OK!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "NT", "hyoo14/NucletideTransformer_PD", 1000, 32, "sequence" #NT       OK!!!!

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "GROVER", "hyoo14/GROVER_PD", 512, 32, "sequence" #GROVER                OK!!!!
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "DNABERT2", "hyoo14/DNABERT2_PD", 510, 32, "sequence" #DNABERT2         OK!!!!
# attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "NT", "hyoo14/NucletideTransformer_PD", 1000, 32, "sequence" #NT          worker



## AMR
task_name = "AMR"
test_file_path = test_dir = "data/df9class_CARD_MEGARes_test_dc.csv"
train_dir = "data/df9class_CARD_MEGARes_train_dc.csv"
valid_dir = "data/df9class_CARD_MEGARes_val_dc.csv"

test_label_name = "Drug Class"

train_df = pd.read_csv(train_dir)
drug_id_to_label = train_df.drop_duplicates().set_index('label')[test_label_name].to_dict()
label_to_id = {v: k for k, v in drug_id_to_label.items()}


attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "GROVER", "hyoo14/GROVER_AMR", 512, 32, "DNA Sequence" #GROVER
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "DNABERT2", "hyoo14/DNABERT2_AMR", 510, 32, "DNA Sequence" #DNABERT2
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "nucl", "NT", "hyoo14/NucletideTransformer_AMR", 1000, 32, "DNA Sequence" #NT

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "GROVER", "hyoo14/GROVER_AMR", 512, 32, "DNA Sequence" #GROVER
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "DNABERT2", "hyoo14/DNABERT2_AMR", 510, 32, "DNA Sequence" #DNABERT2
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "codon", "NT", "hyoo14/NucletideTransformer_AMR", 1000, 32, "DNA Sequence" #NT

#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "GROVER", "hyoo14/GROVER_AMR", 512, 32, "DNA Sequence" #GROVER
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "DNABERT2", "hyoo14/DNABERT2_AMR", 510, 32, "DNA Sequence" #DNABERT2          OK??
#attack_name, model_short_name, model_name, max_len, batch_size, input_name = "bt", "NT", "hyoo14/NucletideTransformer_AMR", 1000, 32, "DNA Sequence" #NT





if model_short_name == "GROVER":
    model_name = "PoetschLab/GROVER"
elif model_short_name == "DNABERT2":
    model_name = "zhihan1996/DNABERT-2-117M"
elif model_short_name == "NT":
    model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"


tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha= 32, lora_dropout=0.1, target_modules= ["query", "value"]
    )


if model_short_name == "DNABERT2":
    # Hugging Face에 로그인합니다.

    hf_model_name = "zhihan1996/DNABERT-2-117M"  # DNABERT-2 원본 모델

    # BertConfig 불러오기 (레이블 수 설정)
    config = BertConfig.from_pretrained(hf_model_name)
    config.num_labels = len(label_to_id)

    # DNABERT-2 모델 불러오기
    base_model3 = AutoModelForSequenceClassification.from_pretrained(hf_model_name, config=config)
    lora_classifier3 = get_peft_model(base_model3, peft_config)
    lora_classifier3.print_trainable_parameters()

    # 1Base Model Freeze (기본 모델 고정)
    for param in lora_classifier3.base_model.parameters():
        param.requires_grad = False  # 기본 모델의 모든 가중치 동결

    #  LoRA 가중치만 학습 가능하게 설정
    for name, param in lora_classifier3.named_parameters():
        if "lora" in name:  # LoRA 관련 가중치만 업데이트
            param.requires_grad = True

    lora_classifier = lora_classifier3
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id)).to(device)


    lora_classifier = get_peft_model(model, peft_config)
    lora_classifier.print_trainable_parameters()
    lora_classifier.to(device)


output_dir = "content_adv"





# 데이터 전처리 함수
def preprocess_data(file_path, max_len):
    """Load and preprocess data from a CSV file."""
    df = pd.read_csv(file_path)
    df[input_name] = df[input_name].str.upper()
    df[input_name] = df[input_name].apply(lambda x: x[:max_len])
    return df







# Dataset 및 DataLoader 생성 함수
def create_dataloader(df, label_to_id, batch_size):
    """Create a tokenized DataLoader from a dataframe."""
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {test_label_name: label_to_id[x[test_label_name]]})
    dataset = Dataset.from_dict({"data": dataset[input_name], 'labels': dataset[test_label_name]})

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["data"], padding=True, truncation=True),
        batched=True,
        remove_columns=["data"]
    )

    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        collate_fn=data_collator,
        batch_size=batch_size
    )
    return dataloader






# Prediction 함수
def predict(model, dataloader, device):
    """Run prediction on the dataloader and return predictions and true labels."""
    model.to(device)
    model.eval()
    all_predictions, all_true_labels, all_logits = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != 'labels'}

            logits = model(**inputs).logits
            predictions = np.argmax(logits.cpu().numpy(), axis=-1)

            all_predictions.extend(predictions)
            all_true_labels.extend(batch['labels'].cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return all_predictions, all_true_labels, np.concatenate(all_logits)

# 평가 및 시각화 함수
def evaluate_and_visualize(true_labels, predictions, label_to_id, output_dir):
    """Evaluate performance metrics and visualize confusion matrix."""
    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    # Confusion Matrix
    id_to_label = {v: k for k, v in label_to_id.items()}
    true_labels_str = [id_to_label[label] for label in true_labels]
    predicted_labels_str = [id_to_label[label] for label in predictions]
    unique_labels = sorted(label_to_id.keys(), key=lambda x: label_to_id[x])

    cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=unique_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # Save image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    now = datetime.now()
    plt.savefig(f'{output_dir}/confusion_matrix_{now.strftime("%Y%m%d_%H%M%S")}.png')

    # Save metrics to file
    with open(f'{output_dir}/results_{now.strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"F1 Score (Macro): {f1:.4f}\n")
        f.write(f"Precision (Macro): {precision:.4f}\n")
        f.write(f"Recall (Macro): {recall:.4f}\n")

from transformers import TrainingArguments

def get_training_args(model_name, output_dir, batch_size=8, learning_rate=5e-4, num_train_epochs=2, max_steps=1000, seed=42):
    """
    모델 학습을 위한 TrainingArguments를 반환하는 함수

    Args:
        model_name (str): 사용할 모델 이름
        output_dir (str): 모델이 저장될 디렉토리
        batch_size (int, optional): 훈련 배치 크기. 기본값 8.
        learning_rate (float, optional): 학습률. 기본값 5e-4.
        num_train_epochs (int, optional): 학습 epoch 수. 기본값 2.
        max_steps (int, optional): 최대 학습 스텝 수. 기본값 1000.
        seed (int, optional): 랜덤 시드 값. 기본값 42.

    Returns:
        TrainingArguments 객체
    """
    return TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=64,
        num_train_epochs=num_train_epochs,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps=max_steps,
        seed=seed
    )

# 사용 예시
output_dir = "dnabert-2-finetuned"
training_args = get_training_args(model_name="dnabert-2", output_dir=output_dir)


import torch
import random
import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score

def set_seed(seed=42):
    """재현성을 위해 랜덤 시드 설정"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics_macro_f1(eval_pred):
    """F1 매크로 스코어 계산"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    f1_macro = f1_score(references, predictions, average="macro")
    return {'f1_macro': f1_macro}

def get_training_args(output_dir, batch_size=8, learning_rate=5e-4, num_train_epochs=2, max_steps=1000, seed=42):
    """TrainingArguments 설정"""
    return TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=64,
        num_train_epochs=num_train_epochs,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps=max_steps,
        seed=seed,
        report_to="none"  # wandb 비활성화
    )

def train_model(model, train_dataset, eval_dataset, tokenizer, output_dir, batch_size=8, learning_rate=5e-4, num_train_epochs=2, max_steps=1000, seed=42):
    """모델 학습 함수"""
    set_seed(seed)  # 랜덤 시드 설정

    # TrainingArguments 생성
    training_args = get_training_args(output_dir, batch_size, learning_rate, num_train_epochs, max_steps, seed)

    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_macro_f1,
    )

    # 모델 학습
    train_results = trainer.train()
    return trainer, train_results


# ATTACK


# Define mutation functions
def nucleotide_mutation(sequence, mutation_rate=0.1):
    """Introduce random mutations into a nucleotide sequence."""
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            sequence[i] = random.choice('ATCG')
    return ''.join(sequence)

def nucleotide_attack(sequences, mutation_rate=0.1, iteration=1):
    """Apply mutations to a list of sequences."""
    mutated_sequences = sequences.copy()  # copy original
    for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: nucleotide_mutation(seq, mutation_rate)
        )
    return mutated_sequences

def codon_mutation(sequence, mutation_rate=0.1):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    for i in range(len(codons)):
        if random.random() < mutation_rate:
            codons[i] = ''.join(random.choices('ATCG', k=3))
    return ''.join(codons)


def codon_attack(sequences, mutation_rate=0.1, iteration=1):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: codon_mutatation(seq, mutation_rate)
        )
  return mutated_sequences




# DNA codon to amino acid translation table
dna_to_aa_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

# Reverse translation table from amino acids to DNA codons
aa_to_dna_table = {v: [k for k in dna_to_aa_table if dna_to_aa_table[k] == v] for v in set(dna_to_aa_table.values())}

def dna_to_aa(sequence):
    return ''.join(dna_to_aa_table.get(sequence[i:i+3], 'X') for i in range(0, len(sequence), 3))

def aa_to_dna(aa_sequence):
    return ''.join(np.random.choice(aa_to_dna_table[aa]) for aa in aa_sequence if aa in aa_to_dna_table)

def back_translation(sequence):
    aa_sequence = dna_to_aa(sequence)
    #print(aa_sequence)
    translated_dna = aa_to_dna(aa_sequence)
    #print(translated_dna)
    return translated_dna

def backtranslation_attack(sequences, mutation_rate=None, iteration=None):
  mutated_sequences = sequences.copy()  # copy original
  for _ in range(iteration):
        mutated_sequences = mutated_sequences.apply(
            lambda seq: back_translation(seq)
        )
  return mutated_sequences


# tokenized dataset 생성
def create_tokenized_dataset(df, label_to_id, batch_size):
    """Create a tokenized DataLoader from a dataframe."""
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {test_label_name: label_to_id[x[test_label_name]]})
    dataset = Dataset.from_dict({"data": dataset[input_name], 'labels': dataset[test_label_name]})

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["data"], padding=True, truncation=True),
        batched=True,
        remove_columns=["data"]
    )

    return tokenized_dataset


# Function to apply mutation only when conflicts exist
def mutate_with_check(seq, not_seq1, not_seq2, not_seq3, not_seq4, mutation_rate=0.1):
    """seq를 perturbation하되, not_seq1과 not_seq2와 같아지지 않도록 보장"""

    if attack_name == "nucl":
        new_seq = nucleotide_mutation(seq, mutation_rate)

        # 동일한 경우에만 다시 mutation (최대 5번 반복)
        retry_count = 0
        while (new_seq in [not_seq1, not_seq2, not_seq3, not_seq4]): # and retry_count < 5:
            new_seq = nucleotide_mutation(seq, mutation_rate)
            retry_count += 1

    elif attack_name == "codon":
        new_seq = codon_mutation(seq, mutation_rate)
        retry_count = 0
        while (new_seq in [not_seq1, not_seq2, not_seq3, not_seq4]):
            new_seq = codon_mutation(seq, mutation_rate)
            retry_count += 1

    elif attack_name == "bt":
        new_seq = back_translation(seq)
        retry_count = 0
        while (new_seq in [not_seq1, not_seq2, not_seq3, not_seq4]):
            new_seq = back_translation(seq)
            retry_count += 1


    # 만약 여전히 같다면 원본 seq 유지 (변경하지 않음)
    return new_seq if new_seq not in [not_seq1, not_seq2, not_seq3, not_seq4] else seq



# make adv train dataset
def make_adv_train_dataset(target_df_dir, not_to_be_df_dir, not_to_be2_df_dir, not_to_be3_df_dir, not_to_be4_df_dir):


    # Load CSV files
    target_df = pd.read_csv(target_df_dir)
    not_to_be_df = pd.read_csv(not_to_be_df_dir)
    not_to_be2_df = pd.read_csv(not_to_be2_df_dir)
    not_to_be3_df = pd.read_csv(not_to_be3_df_dir)
    not_to_be4_df = pd.read_csv(not_to_be4_df_dir)

    # Ensure all DataFrames have the same shape
    assert target_df.shape == not_to_be_df.shape == not_to_be2_df.shape, "❌ CSV 파일들의 크기가 다릅니다!"

    # Apply mutation only to conflicting sequences
    mutated_df = target_df.copy()

    for idx in range(len(target_df)):
        original_seq = target_df[input_name][idx]
        not_seq1 = not_to_be_df[input_name][idx]
        not_seq2 = not_to_be2_df[input_name][idx]
        not_seq3 = not_to_be3_df[input_name][idx]
        not_seq4 = not_to_be4_df[input_name][idx]

        mutated_df[input_name][idx] = mutate_with_check(original_seq, not_seq1, not_seq2, not_seq3, not_seq4, mutation_rate=0.1)

    return mutated_df
    # # Save results
    # mutated_df.to_csv("/content/drive/MyDrive/RDL/prj/data/ADV_TRAIN_test_atk_GROVER_nucl_iter300_rate0.1.csv", index=False)


    
def save_model(lora_classifier, tokenizer, repo_name):
    login("my_key")

    api = HfApi()
    # Hugging Face 리포지토리 생성 (이미 있으면 예외 발생)     try:         api.create_repo(repo_name, private=False)         print(f"Repo '{repo_name}' successfully created.")     except Exception as e:         print(f"Repo '{repo_name}' already exists. Skipping creation.")
    
    # Hugging Face 리포지토리 생성 (이미 있으면 예외 발생)
    try:
        api.create_repo(repo_name, private=False)
        print(f"Repo '{repo_name}' successfully created.")
    except Exception as e:
        print(f"Repo '{repo_name}' already exists. Skipping creation.")


    # 저장소에 모델과 토크나이저 업로드
    lora_classifier.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    if "DNABERT2" in repo_name:
        # 저장 경로 설정
        save_path = "content/lora_state_dict.safetensors"

        # 현재 모델 가중치 저장
        save_file(lora_classifier.state_dict(), save_path)

        api.upload_file(
          path_or_fileobj=save_path,  # 로컬 파일 경로
          path_in_repo="lora_state_dict.safetensors",  # Hugging Face repo 내 저장될 파일명
          repo_id=repo_name
        )




train_way="iter"

# Main 실행에 추가된 부분
if __name__ == "__main__":
    # 원본 테스트 데이터 로드 및 전처리

    target_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter300_rate0.1.csv"
    not_to_be_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter400_rate0.1.csv"
    not_to_be2_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter500_rate0.1.csv"
    not_to_be3_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter100_rate0.1.csv"
    not_to_be4_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter200_rate0.1.csv"

    save_repo_name = f"hyoo14/{model_short_name}_ADV_{task_name}_{attack_name}_iter300"

    adv_train_dataset_save_dir = f"data/ADV_TRAIN_test_atk_{model_short_name}_{attack_name}_iter300_rate0.1.csv"


    mutated_df = make_adv_train_dataset(target_df_dir, not_to_be_df_dir, not_to_be2_df_dir, not_to_be3_df_dir, not_to_be4_df_dir)
    # Save mutated_df
    mutated_df.to_csv(adv_train_dataset_save_dir, index=False)


    # make adv train dataset

    train_df = preprocess_data(train_dir, max_len)
    test_df = preprocess_data(test_dir, max_len)
    valid_df = preprocess_data(valid_dir, max_len)
    mutated_df = preprocess_data(adv_train_dataset_save_dir, max_len)

    mutated_df[test_label_name] = test_df[test_label_name]

    adv_train_df = pd.concat([train_df, mutated_df], axis=0, ignore_index=True)

    # 라벨-아이디 매핑 생성
    print(f"Label-to-ID Mapping: {label_to_id}")

    # DataLoader 생성
    test_dataloader = create_dataloader(test_df, label_to_id, batch_size)
    #test_atk_dataloader = create_dataloader(test_df, label_to_id, batch_size)

    tokenized_datasets_train = create_tokenized_dataset(adv_train_df, label_to_id, batch_size)
    tokenized_datasets_validation = create_tokenized_dataset(valid_df, label_to_id, batch_size)


    print(f"{task_name}_{model_short_name}_{attack_name}_{train_way}")
    # 모델 학습 실행
    output_dir = f"{task_name}_{model_short_name}_{attack_name}_{train_way}"
    trainer, train_results = train_model(
        model=lora_classifier,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_validation,
        tokenizer=tokenizer,
        output_dir=output_dir
    )

    # 저장
    save_model(lora_classifier, tokenizer, save_repo_name)


    # 원본 데이터 예측
    print("Running prediction for original test data...")
    predictions, true_labels, logits = predict(lora_classifier, test_dataloader, device)

    # 평가 및 시각화 (원본 데이터)
    print("\nEvaluation for original test data:")
    evaluate_and_visualize(true_labels, predictions, label_to_id, output_dir)


train_way="ratio"
print(f"{task_name}_{model_short_name}_{attack_name}")
# Main 실행에 추가된 부분
if __name__ == "__main__":
    # 원본 테스트 데이터 로드 및 전처리

    target_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_3.csv"
    not_to_be_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_4.csv"
    not_to_be2_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_5.csv"
    not_to_be3_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_1.csv"
    not_to_be4_df_dir = f"data/test_{task_name}_atk_{model_short_name}_{attack_name}_iter1_rate0_2.csv"

    save_repo_name = f"hyoo14/{model_short_name}_ADV_{task_name}_{attack_name}_ratio0_3"

    adv_train_dataset_save_dir = f"data/ADV_TRAIN_test_atk_{model_short_name}_{attack_name}_iter1_rate0_3.csv"


    mutated_df = make_adv_train_dataset(target_df_dir, not_to_be_df_dir, not_to_be2_df_dir, not_to_be3_df_dir, not_to_be4_df_dir)
    # Save mutated_df
    mutated_df.to_csv(adv_train_dataset_save_dir, index=False)


    # make adv train dataset

    train_df = preprocess_data(train_dir, max_len)
    test_df = preprocess_data(test_dir, max_len)
    valid_df = preprocess_data(valid_dir, max_len)
    mutated_df = preprocess_data(adv_train_dataset_save_dir, max_len)

    mutated_df[test_label_name] = test_df[test_label_name]

    adv_train_df = pd.concat([train_df, mutated_df], axis=0, ignore_index=True)

    # 라벨-아이디 매핑 생성
    print(f"Label-to-ID Mapping: {label_to_id}")

    # DataLoader 생성
    test_dataloader = create_dataloader(test_df, label_to_id, batch_size)
    #test_atk_dataloader = create_dataloader(test_df, label_to_id, batch_size)

    tokenized_datasets_train = create_tokenized_dataset(adv_train_df, label_to_id, batch_size)
    tokenized_datasets_validation = create_tokenized_dataset(valid_df, label_to_id, batch_size)


    print(f"{task_name}_{model_short_name}_{attack_name}_{train_way}")
    # 모델 학습 실행
    output_dir = f"{task_name}_{model_short_name}_{attack_name}_{train_way}"
    trainer, train_results = train_model(
        model=lora_classifier,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_validation,
        tokenizer=tokenizer,
        output_dir=output_dir
    )

    # 저장
    save_model(lora_classifier, tokenizer, save_repo_name)


    # 원본 데이터 예측
    print("Running prediction for original test data...")
    predictions, true_labels, logits = predict(lora_classifier, test_dataloader, device)

    # 평가 및 시각화 (원본 데이터)
    print("\nEvaluation for original test data:")
    evaluate_and_visualize(true_labels, predictions, label_to_id, output_dir)



