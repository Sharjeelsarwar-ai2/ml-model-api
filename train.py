"""
train.py — Train a binary sentiment classifier and save it as one joblib file.

Dataset: UCI Sentiment Labelled Sentences
(https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
3,000 labelled sentences from Amazon, IMDb and Yelp reviews (1 = positive, 0 = negative).

Model: TF-IDF (max 5,000 features) + Logistic Regression.
No PyTorch, no Docker — plain scikit-learn, light enough for any laptop.

Usage:
    python train.py
Output:
    model/sentiment_model.joblib   (dict: {"vectorizer": ..., "model": ..., "meta": {...}})
"""

import io
import os
import urllib.request
import zipfile

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "sentiment_model.joblib")
FILES = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]


def load_data() -> tuple[list[str], list[int]]:
    """Download the UCI zip (once) and parse the three labelled files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "sentiment_labelled_sentences.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("Download complete.")

    texts, labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for fname in FILES:
            member = f"sentiment labelled sentences/{fname}"
            with zf.open(member) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    # Format: "<sentence>\t<0|1>"
                    text, label = line.rsplit("\t", 1)
                    texts.append(text)
                    labels.append(int(label))
    return texts, labels


def main() -> None:
    texts, labels = load_data()
    print(f"Loaded {len(texts)} labelled sentences "
          f"({sum(labels)} positive, {len(labels) - sum(labels)} negative)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_vec))
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, model.predict(X_test_vec),
                                target_names=["negative", "positive"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "meta": {
                "dataset": "UCI Sentiment Labelled Sentences (amazon + imdb + yelp, 3000 rows)",
                "test_accuracy": round(float(acc), 4),
                "vectorizer": "TF-IDF, max_features=5000, ngram_range=(1,2), stop_words=english",
                "model": "LogisticRegression(max_iter=1000, C=1.0)",
                "labels": ["negative", "positive"],
            },
        },
        MODEL_PATH,
    )
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Saved model to {MODEL_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
