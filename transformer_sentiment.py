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

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Text cleaning function (optional)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)       # remove mentions and hashtags
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# Load dataset
df = pd.read_csv("emotion_dataset_raw.csv")
df = df[~df["Emotion"].isin(["disgust", "shame"])]
#df['text'] = df['text'].astype(str).apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Emotion'])

# Save label encoder
with open("transformer_sentiment_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Tokenization
tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>") #tokenizer to convert text to integer sequences, Words not in the top 10,000 will be mapped to <OOV> (out-of-vocab)
tokenizer.fit_on_texts(df['Text']) # word index dictionary

# Save tokenizer
with open("transformer_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['Text'])
X = pad_sequences(sequences, maxlen=256)
y = df['Label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoaders
train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

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

# Initialize model
model = TransformerClassifier(vocab_size=30000, embed_dim=256, nhead=8, hidden_dim=512,
                              num_layers=3, num_classes=len(label_encoder.classes_), max_len=256)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(1000):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/100 - Loss: {total_loss:.4f}")
    if total_loss < 1: 
        break

# Evaluation
model.eval()
all_preds, all_labels, all_confidences = [], [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = probs.max(dim=1).values
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_confidences.extend(confs.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
mean_conf = np.mean(all_confidences)
print("\n--- Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print(f"Mean Confidence: {mean_conf:.4f}")

# Save model
torch.save(model.state_dict(), "transformer_sentiment_model.pt")
print("\nModel and tokenizer saved successfully.")

# --- Predict on combined_data.csv ---
print("\nPredicting on combined_data.csv using trained sentiment model...")

new_df = pd.read_csv('combined_data.csv')
new_df = new_df.dropna(subset=['statement'])

sequences = tokenizer.texts_to_sequences(new_df['statement'])
X_new = pad_sequences(sequences, maxlen=128)

model.eval()
predictions = []
confidences = []

with torch.no_grad():
    for i in range(0, len(X_new), 16):
        batch = torch.tensor(X_new[i:i+16], dtype=torch.long).to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = probs.max(dim=1).values
        predictions.extend(preds.cpu().numpy())
        confidences.extend(confs.cpu().numpy())

decoded_preds = label_encoder.inverse_transform(predictions)

with open('transformer_mental_dataset_predictions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['prediction', 'text'])
    for pred, text in zip(decoded_preds, new_df['statement'].tolist()):
        writer.writerow([pred, text])

print("Prediction file saved: transformer_mental_dataset_predictions.csv")
