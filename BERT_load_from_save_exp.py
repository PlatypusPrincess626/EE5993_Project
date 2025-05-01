# Reload model and label encoder for inference
from transformers import BertForSequenceClassification, BertTokenizer

# Reload model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_CLASSES)
model.load_state_dict(torch.load("bert_sentiment_model.pt"))
model.to(device)
model.eval()

# Reload label encoder
with open("sentiment_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Use tokenizer as before
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

"""
Now predict across the dataset!
"""   
# Load dataset
new_df = pd.read_csv('combined_data.csv')
new_df = new_df.dropna(subset=['statement'])  # Drop rows with missing text
# Encode labels
mental_label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['status'])

mental_dataset = EmotionDataset(df['statement'].tolist(), df['Label'].tolist(), tokenizer, max_length)
mental_loader = DataLoader(mental_dataset, batch_size=batch_size)

all_preds = []
all_labels = []
all_text = []

with torch.no_grad():
    for batch in mental_loader:
        text = batch['text']
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
        all_text.extend(text)

# Save to file
with open('BERT_mental_dataset_predictions.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['prediction', 'label', 'text'])  # header
    for pred, label, text in zip(all_preds, all_labels, all_text):
        writer.writerow([pred, label, text])