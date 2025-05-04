import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, hidden_dim, num_layers, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# Load tokenizers
with open("transformer_tokenizer.pkl", "rb") as f:
    sentiment_tokenizer = pickle.load(f)
with open("transformer_mental_tokenizer.pkl", "rb") as f:
    mental_tokenizer = pickle.load(f)

# Load label encoders
with open("transformer_sentiment_label_encoder.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)
with open("transformer_mental_label_encoder.pkl", "rb") as f:
    mental_encoder = pickle.load(f)

# Model parameters
max_len = 256
vocab_size = 30000
embed_dim = 256
nhead = 4
hidden_dim = 512
num_layers = 3

# Load models
sentiment_model = TransformerClassifier(vocab_size, embed_dim, nhead, hidden_dim, num_layers,
                                  num_classes=len(sentiment_encoder.classes_), max_len=max_len)
sentiment_model.load_state_dict(torch.load("transformer_sentiment_model.pt"))
sentiment_model.to(device)
sentiment_model.eval()

mental_model = TransformerClassifier(vocab_size, embed_dim, nhead, hidden_dim, num_layers,
                            num_classes=len(mental_encoder.classes_), max_len=max_len)
mental_model.load_state_dict(torch.load("transformer_mental_model.pt"))
mental_model.to(device)
mental_model.eval()

# Load test data
df = pd.read_csv('mental_health.csv')
df_small = df.iloc[:100].dropna(subset=['text'])
texts = df_small['text'].tolist()

# Tokenize and pad
sentiment_seq = sentiment_tokenizer.texts_to_sequences(texts)
sentiment_pad = pad_sequences(sentiment_seq, maxlen=max_len)
mental_seq = mental_tokenizer.texts_to_sequences(texts)
mental_pad = pad_sequences(mental_seq, maxlen=max_len)

sentiment_tensor = torch.tensor(sentiment_pad, dtype=torch.long).to(device)
mental_tensor = torch.tensor(mental_pad, dtype=torch.long).to(device)

# Predict
with torch.no_grad():
    s_logits = sentiment_model(sentiment_tensor)
    s_probs = F.softmax(s_logits, dim=1)
    s_conf, s_pred = torch.max(s_probs, dim=1)

    m_logits = mental_model(mental_tensor)
    m_probs = F.softmax(m_logits, dim=1)
    m_conf, m_pred = torch.max(m_probs, dim=1)

# Decode labels
decoded_s = sentiment_encoder.inverse_transform(s_pred.cpu().numpy())
decoded_m = mental_encoder.inverse_transform(m_pred.cpu().numpy())

# Save predictions
df_small["Predicted Sentiment"] = decoded_s
df_small["Sentiment Confidence"] = s_conf.cpu().numpy()
df_small["Predicted Mental"] = decoded_m
df_small["Mental Confidence"] = m_conf.cpu().numpy()
df_small.to_csv("Transformer_full_predictions.csv", index=False)

print("Saved Transformer_full_predictions.csv")
