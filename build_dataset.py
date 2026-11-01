"""
build_dataset.py — Build the combined training CSV for train.py.

Combines:
  1. UCI Sentiment Labelled Sentences (3,000 review sentences, already in
     data/sentiment_labelled_sentences.zip — downloaded once by train.py)
  2. A balanced sample of Sentiment140 tweets (50,000 positive + 50,000 negative)
     — social-media style language the review-only dataset never sees
     ("yay", "passed my exam", "got the job", ...). Chosen by the Step 11
     error analysis: 100,000 tweets + word/char TF-IDF was the best of 24
     candidate configurations on a fixed 2,000-sentence eval set. Note the
     committed CSV is ~6.9 MB (over the original ~5 MB guidance) — the extra
     50,000 tweets buy the last point of accuracy and fix the "Why this always
     happens to me" regression; see Step 11 for the trade-off. It is written
     as 4 part files (data/train_sentences_part1–4.csv, ~1.7 MB each) because
     the GitHub push API rejects single requests over ~7 MB.

The Sentiment140 zip (~80 MB) is NOT needed at deploy/build time: it is
downloaded once by the developer and only the 10,000 sampled tweets are kept.

Usage:
    1. Download https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
       and unzip training.1600000.processed.noemoticon.csv next to this script
       (or pass its path with --tweets).
    2. python build_dataset.py
Output:
    data/train_sentences_part1–4.csv  (UTF-8, columns: text,label — 103,000 rows total)
"""

import argparse
import csv
import io
import os
import random
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
UCI_ZIP = os.path.join(DATA_DIR, "sentiment_labelled_sentences.zip")
UCI_FILES = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
OUT_PARTS = [os.path.join(DATA_DIR, f"train_sentences_part{i}.csv") for i in range(1, 5)]

N_TWEETS_PER_CLASS = 50000
RANDOM_STATE = 7

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_tweet(text: str) -> str:
    """Light cleaning only: strip URLs, @mentions and a leading 'RT '.
    The words themselves are left exactly as the user wrote them."""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = re.sub(r"^RT\s+", "", text)
    return " ".join(text.split()).strip()


def load_uci() -> list[tuple[str, int]]:
    rows = []
    with zipfile.ZipFile(UCI_ZIP) as zf:
        for fname in UCI_FILES:
            member = f"sentiment labelled sentences/{fname}"
            with zf.open(member) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    text, label = line.rsplit("\t", 1)
                    rows.append((text, int(label)))
    return rows


def load_tweets(path: str) -> list[tuple[str, int]]:
    """Read Sentiment140, sample N_TWEETS_PER_CLASS per class (seeded)."""
    rng = random.Random(RANDOM_STATE)
    # Reservoir sampling per class so we never hold 1.6M rows in memory.
    samples: dict[int, list[str]] = {0: [], 1: []}
    seen: dict[int, int] = {0: 0, 1: 0}
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                continue
            polarity = row[0].strip().strip('"')
            if polarity not in ("0", "4"):
                continue
            label = 0 if polarity == "0" else 1
            text = clean_tweet(row[5])
            if len(text) < 3:
                continue
            seen[label] += 1
            bucket = samples[label]
            if len(bucket) < N_TWEETS_PER_CLASS:
                bucket.append(text)
            else:
                j = rng.randrange(seen[label])
                if j < N_TWEETS_PER_CLASS:
                    bucket[j] = text
    return [(t, 0) for t in samples[0]] + [(t, 1) for t in samples[1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tweets",
                        default=os.path.join(HERE, "training.1600000.processed.noemoticon.csv"),
                        help="Path to the unzipped Sentiment140 training CSV")
    args = parser.parse_args()

    uci = load_uci()
    print(f"UCI reviews: {len(uci)} ({sum(l for _, l in uci)} positive)")

    tweets = load_tweets(args.tweets)
    n_pos = sum(l for _, l in tweets)
    print(f"Sentiment140 sample: {len(tweets)} ({n_pos} positive, {len(tweets) - n_pos} negative)")

    rows = uci + tweets
    rng = random.Random(RANDOM_STATE)
    rng.shuffle(rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    per = (len(rows) + len(OUT_PARTS) - 1) // len(OUT_PARTS)
    for i, out in enumerate(OUT_PARTS):
        with open(out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["text", "label"])
            writer.writerows(rows[i * per:(i + 1) * per])
    total_mb = sum(os.path.getsize(p) for p in OUT_PARTS) / (1024 * 1024)
    print(f"Wrote {len(OUT_PARTS)} part files: {len(rows)} rows "
          f"({sum(l for _, l in rows)} positive), {total_mb:.2f} MB total")


if __name__ == "__main__":
    main()
