# Transformer Sentiment Model

This repository contains a custom Transformer-based model for performing sentiment (emotion) classification on social media text data. The model is trained from scratch on the `emotion_dataset_raw.csv` dataset.

---

## 🔧 Features

* Custom Transformer Encoder model (no pretraining)
* Multi-class classification (emotions like joy, sadness, anger, etc.)
* Keras tokenizer (TF backend) with PyTorch model training
* Lightweight architecture with tunable parameters

---

## 📁 Dataset

* **Source**: `emotion_dataset_raw.csv`
* **Columns**: `text`, `emotion`
* **Size**: \~35,000 rows

---

## 🚀 How to Run

1. **Install dependencies**:

   ```bash
   pip install torch scikit-learn pandas numpy tensorflow
   ```

2. **Run the training script**:

   ```bash
   python transformer_sentiment.py
   ```

3. **Outputs**:

   * `transformer_sentiment_model.pt` – trained model
   * `transformer_tokenizer.pkl` – saved tokenizer
   * `transformer_sentiment_label_encoder.pkl` – label encoder

---

## 📊 Evaluation

Evaluation is printed after training:

* Accuracy
* F1 Score (macro)
* Mean Confidence

---

## 📈 Planned Improvements

See `transformer_sentiment_improvements.txt` for future updates, including:

* Increased vocabulary size
* Text normalization and cleaning
* Longer training with early stopping
* Dropout and weight decay for generalization

---

## 🧠 Credits

Developed as part of the AI-Driven Sentiment Analysis of Social Media project.

---

## 📄 License

MIT License (or specify your own).

---

## 📬 Contact

For questions, contact the project contributors or open an issue in this repository.
