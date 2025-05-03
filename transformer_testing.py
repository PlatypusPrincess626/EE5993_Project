import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models and tokenizer
from transformer_sentiment import TransformerClassifier as SentimentModel
from transformer_mental import TransformerClassifier as MentalModel

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
max_len = 128
vocab_size = 10000
embed_dim = 128
nhead = 4
hidden_dim = 256
num_layers = 2

# Load models
sentiment_model = SentimentModel(vocab_size, embed_dim, nhead, hidden_dim, num_layers,
                                  num_classes=len(sentiment_encoder.classes_), max_len=max_len)
sentiment_model.load_state_dict(torch.load("transformer_sentiment_model.pt"))
sentiment_model.to(device)
sentiment_model.eval()

mental_model = MentalModel(vocab_size, embed_dim, nhead, hidden_dim, num_layers,
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
