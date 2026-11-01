"""
train.py — Train a binary sentiment classifier and save it as one joblib file.

Dataset: data/train_sentences_part1–4.csv (103,000 labelled sentences, UTF-8,
built by build_dataset.py and committed to the repo — no download at
train/deploy time). The data is split into 4 part files (~1.7 MB each)
because the GitHub push API rejects single requests over ~7 MB.
  - UCI Sentiment Labelled Sentences (3,000 Amazon/IMDb/Yelp reviews,
    1,500 positive / 1,500 negative)
  - Sentiment140 tweets, balanced sample (100,000 tweets,
    50,000 positive / 50,000 negative) — social-media style language the
    review-only dataset never sees ("yay", "passed my exam", "got the job").

Evaluation: a FIXED 2,000-sentence eval set (data/eval_sentences.csv,
stratified split of the original 13k-row data, seed 123) is held out of
training — train.py trains on the CSV minus those rows and reports accuracy
on the eval set. The eval set never changes, so accuracy numbers across
configurations are directly comparable (see the build report, Step 11).

Model: word TF-IDF (10,000 features, 1–2 grams) + char TF-IDF (10,000
features, char_wb 3–5 grams) combined, + Logistic Regression (C=2.0).
The char features are what fixed the tweet errors: they catch slang,
stretched words ("looove", "sooo") and typos that word features miss.
No PyTorch, no Docker — plain scikit-learn, light enough for any laptop.

Usage:
    python train.py
Output:
    model/sentiment_model.joblib   (dict: {"vectorizer": ..., "model": ..., "meta": {...}})
"""

import csv
import glob
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PARTS = sorted(glob.glob(os.path.join(HERE, "data", "train_sentences_part*.csv")))
EVAL_CSV = os.path.join(HERE, "data", "eval_sentences.csv")
MODEL_PATH = os.path.join(HERE, "model", "sentiment_model.joblib")


def norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_rows(path: str) -> list[tuple[str, int, str]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rows.append((text, int(row["label"]), row.get("source", "")))
    return rows


def main() -> None:
    if not DATA_PARTS:
        raise SystemExit(
            "Training data not found at data/train_sentences_part*.csv. "
            "Run `python build_dataset.py` first."
        )
    if not os.path.exists(EVAL_CSV):
        raise SystemExit(f"Eval set not found at {EVAL_CSV}.")

    all_rows = []
    for part in DATA_PARTS:
        all_rows.extend(load_rows(part))
    eval_rows = load_rows(EVAL_CSV)
    eval_norms = {norm(t) for t, _, _ in eval_rows}
    train_rows = [(t, l) for t, l, _ in all_rows if norm(t) not in eval_norms]
    print(f"Loaded {len(all_rows)} labelled sentences "
          f"({sum(l for _, l, _ in all_rows)} positive)")
    print(f"Fixed eval set: {len(eval_rows)} sentences "
          f"({sum(l for _, l, _ in eval_rows)} positive) — held out of training")
    print(f"Training on {len(train_rows)} rows "
          f"({sum(l for _, l in train_rows)} positive)")

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 max_features=10000)),
    ])
    X_train = vectorizer.fit_transform([t for t, _ in train_rows])
    X_eval = vectorizer.transform([t for t, _, _ in eval_rows])

    model = LogisticRegression(max_iter=1000, C=2.0)
    model.fit(X_train, [l for _, l in train_rows])

    y_true = [l for _, l, _ in eval_rows]
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_true, y_pred)
    print(f"\nEval accuracy (fixed 2,000-sentence set): {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
    for src in ("review", "tweet"):
        idx = [i for i, (_, _, s) in enumerate(eval_rows) if s == src]
        src_acc = accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx])
        print(f"  {src} accuracy: {src_acc:.4f} (n={len(idx)})")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "meta": {
                "dataset": "UCI Sentiment Labelled Sentences (3,000 reviews) + "
                           "Sentiment140 sample (100,000 tweets), 103,000 rows total",
                "eval_set": "data/eval_sentences.csv — fixed 2,000 sentences, "
                            "seed 123, held out of training",
                "test_accuracy": round(float(acc), 4),
                "vectorizer": "FeatureUnion: word TF-IDF max_features=10000, "
                              "ngram_range=(1,2), stop_words=english + char_wb "
                              "TF-IDF max_features=10000, ngram_range=(3,5)",
                "model": "LogisticRegression(max_iter=1000, C=2.0)",
                "labels": ["negative", "positive"],
            },
        },
        MODEL_PATH,
    )
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Saved model to {MODEL_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
