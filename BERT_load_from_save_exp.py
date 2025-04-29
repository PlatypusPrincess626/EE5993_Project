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