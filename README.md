# 🛡️ AI Spam Email Detector

A beginner-friendly NLP project that classifies SMS/email messages as **spam or ham** using TF-IDF vectorization and a Naive Bayes classifier, served through a Streamlit web app.

## 📁 Project Structure
```
spam-detector/
├── data/
│   └── spam.csv          ← Download from UCI (link below)
├── models/
│   └── spam_model.pkl    ← Auto-generated after training
├── src/
│   ├── train.py          ← Train & save the model
│   └── predict.py        ← Load model & predict
├── app.py                ← Streamlit web interface
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download the **UCI SMS Spam Collection** dataset:
👉 https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Save the file as `data/spam.csv`.

### 3. Train the model
```bash
python src/train.py
```
Expected output: ~97% accuracy.

### 4. Launch the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.


## 💡 Ideas to Extend This
- Swap Naive Bayes for Logistic Regression or SVM and compare results
- Try a pre-trained transformer (e.g. DistilBERT) for higher accuracy
- Add email header features (sender domain, subject line)
- Deploy to Hugging Face Spaces for a live public demo
