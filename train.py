import os
from tqdm import tqdm
import random
import torch
import pandas as pd
import sklearn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    pipeline,
    logging,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig
)
"C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/"
train_dataframe = pd.DataFrame(pd.read_csv("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/train_essays.csv"))
train_dataframe1 = pd.DataFrame(pd.read_csv("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/Training_Essay_Data.csv"))
train_dataframe2 = pd.DataFrame(pd.read_csv("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/Wikipedia.csv"))
train_dataframe3 = pd.DataFrame(pd.read_csv("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/data_set.csv"))
train_dataframe4 = pd.DataFrame(pd.read_csv("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/train_essays_RDizzl3_seven_v2.csv"))
train_dataframe5 = pd.DataFrame(pd.read_csv("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/dataset/dataset.csv"))

tmp_train_dataframe = train_dataframe.drop(columns=["id", "prompt_id"]).rename(columns={"generated": "labels"})
tmp_train_dataframe.sample(frac=1).reset_index(drop=True)

dataset_length = len(tmp_train_dataframe)

tmp_test_dataframe = tmp_train_dataframe.iloc[int(dataset_length * 0.8):]
tmp_train_dataframe = tmp_train_dataframe.iloc[:int(dataset_length * 0.8)]


tmp_train_dataframe1 = train_dataframe1.rename(columns={"generated": "labels"})
tmp_train_dataframe2 = train_dataframe2.drop(columns=["Human"]).rename(columns={"Text":"text", "AI": "labels"})
tmp_train_dataframe3 = train_dataframe3.drop(columns=["title", "ai_generated"]).rename(columns={"abstract":"text", "is_ai_generated": "labels"})
tmp_train_dataframe4 = train_dataframe4.rename(columns={"label": "labels"})
tmp_train_dataframe5 = train_dataframe5.drop(columns=["Unnamed: 0"]).rename(columns={"class": "labels"})
tmp_train_dataframe5["labels"] = tmp_train_dataframe5["labels"].apply(lambda x: 1 if x == "AI-Generated-Text" else 0)

def concat_dataframe(*dataframes):
    result_dataframe = pd.concat(dataframes, axis=0)
    return result_dataframe

train_essays = concat_dataframe(
    tmp_train_dataframe,
    tmp_train_dataframe1,
    tmp_train_dataframe4,
)

# 特定の文字を取り除く
target_word = "\n"
train_essays['text'] =train_essays['text'].str.replace(target_word, '')

from tqdm import tqdm

def max_token_count(dataframe, model_name="roberta-base"):
    # トークナイザーの読み込み
    token_counts_list = []
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 1027

    # トークン数を計算してリストに保存

    for text in tqdm(dataframe["text"]):
      token_counts_list.append(len(tokenizer.encode(text)))

    return max(token_counts_list), token_counts_list

max_seq_length, token_counts_list = max_token_count(train_essays)

lower_max_length_list = [count <= 1024 for count in token_counts_list]

filterd_dataframe = train_essays[lower_max_length_list]

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe['text'].iloc[idx]
        label = self.dataframe['labels'].iloc[idx]

        # Tokenize and encode the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Extract input_ids, attention_mask, and convert label to tensor
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label_tensor = torch.tensor(label)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor
        }
    
# データを訓練用と評価用に分割
train_dataframe, eval_dataframe = train_test_split(filterd_dataframe, test_size=0.1, random_state=42)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer.model_max_length = 1027
batch_size = 2
gradient_accumulation_size = 64
check_steps = gradient_accumulation_size / batch_size

train_dataset = CustomDataset(train_dataframe, tokenizer)
eval_dataset = CustomDataset(eval_dataframe, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

config = RobertaConfig(
    attention_probs_dropout_prob=0.1,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.2,
    intermediate_size=3072,
    max_position_embeddings=1027,
    num_attention_heads=12,
    position_embedding_type="absolute",
    type_vocab_size=1,
    pad_token_id=1,
    num_hidden_layers=12,
    output_attentions=False,
    vocab_size=50265

)

model = RobertaForSequenceClassification(config)

class EarlyStopping:
    def __init__(self, patience):
        self.check_count = 0
        self.patience = patience

    def checkCount(self, _bool):
        if _bool:
            self.check_count = 0
        else:
            self.check_count += 1

        if self.check_count == self.patience:
            return 0
        else:
            return None
        
patience = 5
early_stopping = EarlyStopping(patience=patience)

device = torch.device("cuda")

total_samples = len(train_dataset)
class_0_samples = len(train_dataframe[train_dataframe["labels"] == 0])
class_1_samples = len(train_dataframe[train_dataframe["labels"] == 1])

# クラスごとの重みを計算する。
weight_class_0 = total_samples / class_0_samples
weight_class_1 = total_samples / class_1_samples

# 重みをテンソルに変換
weights = torch.tensor([weight_class_0, weight_class_1]).to(device)


criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
model = model.to(device)

def compute_metrics(label_ids, predictions):
    # AUC-ROCの計算
    probabilities = F.softmax(predictions, dim=-1).cpu().detach().numpy()
    auc_roc = roc_auc_score(label_ids.cpu().detach().numpy(), probabilities[:, 1])  # 2クラス分類の場合、クラス1の確率を使います

    return auc_roc

def train(model, train_dataloader, optimizer, criterion, check_steps, device="cuda"):
    epoch_loss = 0.0
    sum_accumulation_loss = 0.0

    #epoch_auc_roc = 0.0

    model.train()

    for batch, data in tqdm(enumerate(train_dataloader)):
        input_ids =data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels =  data["labels"].to(device)


        outputs = model(input_ids, attention_mask=attention_mask)

        loss = criterion(outputs.logits, labels)
        accumulation_loss = loss / check_steps
        accumulation_loss.backward()
        sum_accumulation_loss += accumulation_loss.item()

        #auc_roc = compute_metrics(labels, outputs.logits)

        if (batch + 1) % check_steps == 0:

            optimizer.step()
            optimizer.zero_grad()

            print()
            print("Gradient Accumulation")
            print(f"Step: Train Iteration: {batch + 1}/{len(train_dataloader)} loss: {sum_accumulation_loss}")
            sum_accumulation_loss = 0.0

        epoch_loss += loss.item()
        #epoch_auc_roc += auc_roc

        print()
        #print(f"Step: Train Iteration: {batch + 1}/{len(train_dataloader)} loss: {loss.item()}")
        #print(f"Step: Train Iteration: {batch + 1} loss: {loss.item()}, auc_roc: {auc_roc.item()}")

    return epoch_loss / (batch + 1)#, epoch_auc_roc / (batch + 1)

def evaluate(model, eval_dataloader, criterion, device="cuda"):
    epoch_loss = 0.0
    #epoch_auc_roc = 0.0

    model.eval()

    with torch.no_grad():
        for batch, data in tqdm(enumerate(eval_dataloader)):
            input_ids =data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels =  data["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs.logits, labels)
            #auc_roc = compute_metrics(outputs.logits, labels)

            epoch_loss += loss.item()
            #epoch_auc_roc += auc_roc

            print()
            print(f"Step: Eval Iteration: {batch + 1}/{len(eval_dataloader)} loss: {loss.item()}")
            #print(f"Step: Eval Iteration: {batch + 1} loss: {loss.item()}, auc_roc: {auc_roc.item()}")

    return epoch_loss / (batch + 1)#, epoch_auc_roc / (batch + 1)

train_loss_list = []
#train_auc_roc_list = []

eval_loss_list = []
#eval_auc_roc_list = []

epochs = 20
min_loss = 1000
model.save_pretrained("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/roberta_classification_model")
tokenizer.save_pretrained("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/roberta_classification_model")
for epoch in range(epochs):

    epoch_train_loss = train(model, train_dataloader, optimizer, criterion, check_steps)
    print()
    print("-"*100)
    print(f"Step: Train EarlyStopping: {early_stopping.check_count}/{early_stopping.patience} Epoch: {epoch + 1}/{epochs}, loss: {epoch_train_loss}")
    #print(f"Step: Train Epoch: {epoch + 1}/{epochs}, loss: {epoch_train_loss}, auc_roc: {epoch_train_auc_roc}")
    print("-"*100)

    epoch_eval_loss = evaluate(model, eval_dataloader, criterion)
    print()
    print("-"*100)
    print(f"Step: Eval EarlyStopping: {early_stopping.check_count}/{early_stopping.patience} Epoch: {epoch + 1}/{epochs}, loss: {epoch_eval_loss}")
    #print(f"Step: Eval Epoch: {epoch + 1}/{epochs}, loss: {epoch_eval_loss}, auc_roc: {epoch_eval_auc_roc}")
    print("-"*100)

    train_loss_list.append(epoch_train_loss)
    #train_auc_roc_list.append(epoch_train_auc_roc)

    eval_loss_list.append(epoch_eval_loss)
    #eval_auc_roc_list.append(epoch_eval_auc_roc)

    if epoch_eval_loss < min_loss:
        min_loss = epoch_eval_loss
        model.save_pretrained("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/roberta_classification_model")
        tokenizer.save_pretrained("C:/Users/yuhei/VScode/python/LLM_Detect_AI_Generated_Text/roberta_classification_model")
        early_stopping.checkCount(True)

    else:
      check = early_stopping.checkCount(False)
      if check == 0:
        print("Early Stopping !")
        break
