"""
train.py — Train a binary sentiment classifier and save it as one joblib file.

Dataset: data/train_sentences.csv (13,000 labelled sentences, UTF-8, built by
build_dataset.py and committed to the repo — no download at train/deploy time):
  - UCI Sentiment Labelled Sentences (3,000 Amazon/IMDb/Yelp reviews,
    1,500 positive / 1,500 negative)
  - Sentiment140 tweets, balanced sample (10,000 tweets,
    5,000 positive / 5,000 negative) — social-media style language the
    review-only dataset never sees ("yay", "passed my exam", "got the job").

Model: TF-IDF (max 5,000 features) + Logistic Regression.
No PyTorch, no Docker — plain scikit-learn, light enough for any laptop.

Usage:
    python train.py
Output:
    model/sentiment_model.joblib   (dict: {"vectorizer": ..., "model": ..., "meta": {...}})
"""

import csv
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data", "train_sentences.csv")
MODEL_PATH = os.path.join(HERE, "model", "sentiment_model.joblib")


def load_data() -> tuple[list[str], list[int]]:
    """Read the committed training CSV (text,label). No download needed."""
    if not os.path.exists(DATA_CSV):
        raise SystemExit(
            f"Training data not found at {DATA_CSV}. Run `python build_dataset.py` first."
        )
    texts, labels = [], []
    with open(DATA_CSV, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            labels.append(int(row["label"]))
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
                "dataset": "UCI Sentiment Labelled Sentences (3,000 reviews) + "
                           "Sentiment140 sample (10,000 tweets), 13,000 rows total",
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
