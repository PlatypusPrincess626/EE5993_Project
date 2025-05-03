import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import GPT2Config
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle


class GPT2ForClassification(GPT2PreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls_token = last_hidden_state[:, -1, :]  # Use last token representation
        logits = self.classifier(self.dropout(cls_token))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default

# Use tokenizer as before
config = GPT2Config.from_pretrained('gpt2')
config.pad_token_id = tokenizer.pad_token_id
sentiment_model = GPT2ForClassification(config, num_labels=6)
# Load weights
sentiment_model.load_state_dict(torch.load("gpt2_sentiment_model.pt"))
sentiment_model.to(device)
with open("gpt2_sentiment_label_encoder.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)
    
# Use tokenizer as before
config = GPT2Config.from_pretrained('gpt2')
config.pad_token_id = tokenizer.pad_token_id
mental_model = GPT2ForClassification(config, num_labels=7)
# Load weights
mental_model.load_state_dict(torch.load("gpt2_mental_model.pt"))
mental_model.to(device)
with open("gpt2_mental_label_encoder.pkl", "rb") as f:
    mental_encoder = pickle.load(f)

# Load dataset
df = pd.read_csv('mental_health.csv')
df_small = df.iloc[:100]
texts = df_small["text"].tolist()

encode = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
encode = {key: val.to(device) for key, val in encode.items()}

with torch.no_grad():
    sentiment_outputs = sentiment_model(**encode)
    logits = sentiment_outputs['logits']
    probs = F.softmax(logits, dim=1)
    sentiment_confidence, sentiment_predicted = torch.max(probs, dim=1)

    mental_outputs = mental_model(**encode)
    logits = mental_outputs['logits']
    probs = F.softmax(logits, dim=1)
    mental_confidence, mental_predicted = torch.max(probs, dim=1)

# Append predictions and confidence to DataFrame
df_small["Predicted Sentiment"] = sentiment_predicted.cpu().numpy()
df_small["Sentiment Confidence"] = sentiment_confidence.cpu().numpy()
df_small["Predicted Mental"] = mental_predicted.cpu().numpy()
df_small["Mental Confidence"] = mental_confidence.cpu().numpy()

# Save to CSV
df_small.to_csv("GPT_full_predictions.csv", index=False)
print("Saved to mental_health_predictions.csv")


