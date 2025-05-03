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
import pickle

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv('combined_data.csv')
df = df.dropna(subset=['statement'])  # Drop rows with missing text

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['status'])

# Get the mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Print it
print(label_mapping)

# Train/Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['statement'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
)

# Parameters
num_classes = len(label_encoder.classes_)
batch_size = 16
epochs = 10
max_length = 256

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model.to(device)

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Create datasets and dataloaders
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
model.train()
for epoch in range(epochs):
    print(f"{epoch+1}/{epochs}")
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1)
        confidences = probs.max(dim=-1).values

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
mean_confidence = np.mean(all_confidences)

print("\n--- Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print(f"Mean Confidence: {mean_confidence:.4f}")

# Map predicted labels back to original emotion strings
predicted_emotions = label_encoder.inverse_transform(all_preds)
true_emotions = label_encoder.inverse_transform(all_labels)

# Example predictions
for i in range(5):
    print(f"True: {true_emotions[i]} - Predicted: {predicted_emotions[i]} - Confidence: {all_confidences[i]:.2f}")
    
# Save model weights and label encoder after training
model_path = "BERT_mental_model.pt"
torch.save(model.state_dict(), model_path)

# Save label encoder to reuse label mapping
with open("BERT_mental_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print(f"\nModel saved to {model_path}")