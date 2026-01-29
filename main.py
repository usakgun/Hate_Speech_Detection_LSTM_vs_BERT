import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW 
from transformers import BertTokenizer, BertForSequenceClassification
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    return text

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'labeled_data.csv')
df = pd.read_csv(file_path)

df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)
df['clean_text'] = df['tweet'].apply(clean_text)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word2idx = {}
index = 1
for text in X_train_raw:
    for word in text.split():
        if word not in word2idx:
            word2idx[word] = index
            index += 1
word2idx['<PAD>'] = 0
vocab_size = len(word2idx) + 1

def encode_text(text_list):
    encoded = []
    for text in text_list:
        tokens = [word2idx.get(w, 0) for w in text.split()]
        if len(tokens) < 50:
            tokens += [0] * (50 - len(tokens))
        else:
            tokens = tokens[:50]
        encoded.append(tokens)
    return torch.tensor(encoded)

X_train_lstm = encode_text(X_train_raw)
X_test_lstm = encode_text(X_test_raw)

y_train_tensor = torch.tensor(y_train.tolist())
y_test_tensor = torch.tensor(y_test.tolist())

train_loader_lstm = DataLoader(list(zip(X_train_lstm, y_train_tensor)), batch_size=32, shuffle=True)
test_loader_lstm = DataLoader(list(zip(X_test_lstm, y_test_tensor)), batch_size=32)

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 128, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(128 * 2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        hidden = torch.max(lstm_out, dim=1)[0]
        return self.fc(self.dropout(hidden))

lstm_model = BiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

print("Starting LSTM Training...")
start_time = time.time()

for epoch in range(2):
    lstm_model.train()
    total_loss = 0
    for inputs, labels in train_loader_lstm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_lstm.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_lstm.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader_lstm)}")

lstm_train_time = time.time() - start_time
print(f"LSTM Training Time: {lstm_train_time}")

lstm_model.eval()
lstm_preds = []
lstm_true = []
start_inf = time.time()

with torch.no_grad():
    for inputs, labels in test_loader_lstm:
        inputs = inputs.to(device)
        outputs = lstm_model(inputs)
        preds = torch.argmax(outputs, dim=1)
        lstm_preds.extend(preds.cpu().numpy())
        lstm_true.extend(labels.numpy())

lstm_inf_time = (time.time() - start_inf) / len(y_test)
print("LSTM Results")
print(classification_report(lstm_true, lstm_preds))
print(f"Inference Time per sample: {lstm_inf_time}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset_bert = BertDataset(X_train_raw.tolist(), y_train.tolist())
test_dataset_bert = BertDataset(X_test_raw.tolist(), y_test.tolist())

train_loader_bert = DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=16)

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

for name, param in bert_model.bert.named_parameters():
    if 'encoder.layer.10' in name or 'encoder.layer.11' in name or 'pooler' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

bert_model = bert_model.to(device)
optimizer_bert = AdamW(bert_model.parameters(), lr=2e-5)

print("Starting BERT Training...")
start_time_bert = time.time()

for epoch in range(2):
    bert_model.train()
    total_loss = 0
    for batch in train_loader_bert:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer_bert.zero_grad()
        outputs = bert_model(input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_bert.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader_bert)}")

bert_train_time = time.time() - start_time_bert
print(f"BERT Training Time: {bert_train_time}")

bert_model.eval()
bert_preds = []
bert_true = []
start_inf_bert = time.time()

with torch.no_grad():
    for batch in test_loader_bert:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = bert_model(input_ids, attention_mask=mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        bert_preds.extend(preds.cpu().numpy())
        bert_true.extend(labels.cpu().numpy())

bert_inf_time = (time.time() - start_inf_bert) / len(y_test)
print("BERT Results")
print(classification_report(bert_true, bert_preds))
print(f"Inference Time per sample: {bert_inf_time}")

print("\n" + "="*50)
print("ERROR ANALYSIS: Qualitative Inspection of Misclassified Tweets")
print("="*50)

bert_misclassified_indices = [i for i, (p, t) in enumerate(zip(bert_preds, bert_true)) if p != t]

print(f"Displaying 5 random misclassified examples out of {len(bert_misclassified_indices)} total errors:\n")

for i in bert_misclassified_indices[:5]:
    text = X_test_raw.iloc[i]
    
    true_label_str = "Hate Speech" if bert_true[i] == 1 else "Non-Hate"
    pred_label_str = "Hate Speech" if bert_preds[i] == 1 else "Non-Hate"
    
    print(f"Tweet: {text}")
    print(f"True Label: {true_label_str} | Predicted: {pred_label_str}")
    print("-" * 40)