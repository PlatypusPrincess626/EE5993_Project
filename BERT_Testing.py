import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import pickle

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define model again with correct number of labels
sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=8 )
# Load weights
sentiment_model.load_state_dict(torch.load("bert_sentiment_model.pt"))
sentiment_model.to(device)
# Load encoder
with open("sentiment_label_encoder.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)

# Define model again with correct number of labels
mental_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=7 )
# Load weights
mental_model.load_state_dict(torch.load("BERT_mental_model.pt"))
mental_model.to(device)
# Load encoder
with open("BERT_mental_label_encoder.pkl", "rb") as f:
    mental_encoder = pickle.load(f)

# Use tokenizer as before
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
df = pd.read_csv('mental_health.csv')
texts = df["text"].tolist()
sentiment_encode = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
mental_encode = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    sentiment_outputs = sentiment_model(**sentiment_encode)
    logits = sentiment_outputs.logits
    probs = F.softmax(logits, dim=1)
    sentiment_confidence, sentiment_predicted = torch.max(probs, dim=1)

    mental_outputs = mental_model(**mental_encode)
    logits = mental_outputs.logits
    probs = F.softmax(logits, dim=1)
    mental_confidence, mental_predicted = torch.max(probs, dim=1)

decoded_sentiment = sentiment_encoder.inverse_transform(sentiment_predicted.numpy())
decoded_mental = mental_encoder.inverse_transform(mental_predicted.numpy())

# Append predictions and confidence to DataFrame
df["Predicted Sentiment"] = sentiment_predicted.numpy()
df["Decoded Sentiment"] = decoded_sentiment
df["Sentiment Confidence"] = sentiment_confidence.numpy()

df["Predicted Mental"] = mental_predicted.numpy()
df["Decoded Mental"] = decoded_mental
df["Mental Confidence"] = mental_confidence.numpy()

# Save to CSV
df.to_csv("BERT_full_predictions.csv", index=False)
print("Saved to mental_health_predictions.csv")



