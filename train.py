"""
train.py — Train and save the spam classifier model.
Run this once before launching the app: python src/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download("stopwords", quiet=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
def load_data(path: str = "data/spam.csv") -> pd.DataFrame:
    """
    Load the UCI SMS Spam Collection dataset.
    Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    Save the file as data/spam.csv
    """
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"✅ Loaded {len(df)} messages  |  Spam: {df['label_num'].sum()}  Ham: {(df['label_num']==0).sum()}")
    return df


# ── 2. Text Preprocessing ─────────────────────────────────────────────────────
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)          # remove punctuation / digits
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


# ── 3. Build & Train Pipeline ─────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess,
            max_features=5000,
            ngram_range=(1, 2),   # unigrams + bigrams
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def train(data_path: str = "data/spam.csv", model_path: str = "models/spam_model.pkl"):
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # ── Save ──
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"💾 Model saved to {model_path}")
    return pipeline


if __name__ == "__main__":
    train()
