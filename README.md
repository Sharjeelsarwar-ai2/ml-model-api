# ML Model as an API — Sentiment Classifier

Train a sentiment classifier, then serve it as a live REST API anyone can send
requests to. **No Docker, no PyTorch** — plain scikit-learn, FastAPI and
`uvicorn`, light enough for any laptop.

```
Dataset → Train → Saved model → REST API → Live requests
```

## What it does

- `POST /predict` — send `{"text": "..."}` and get back
  `{"label": "positive", "score": 0.94}` (score = probability of the predicted label)
- `GET /demo` — one demo page with two tabs: **Text sentiment** (type a
  sentence, get a label + confidence + plain-words feeling) and **Face emotion**
  (camera or photo upload, 7 expressions — happy, sad, angry, scared,
  disgusted, surprised, neutral — 100% client-side with face-api.js, nothing
  is sent to the server). Tab state lives in the URL hash: `/demo#text` and
  `/demo#face` link straight to each tool. The front-camera preview is
  mirrored, like the phone's camera app (the face box is mirrored with it,
  so it stays on the face); photos are never mirrored.
- `GET /face` — redirects to `/demo#face`, so the old face link still works.
- `GET /` opens the demo page in a browser (it redirects to `/demo`)

## The model

- **Dataset:** 13,000 labelled sentences — UCI Sentiment Labelled Sentences
  (3,000 Amazon/IMDb/Yelp reviews, 1,500 positive / 1,500 negative) **plus**
  a balanced sample of Sentiment140 tweets (10,000 tweets, 5,000 positive /
  5,000 negative). The review-only model misclassified everyday happy news
  ("I passed my course as top performer yay" → negative 0.56), because
  achievement / life-update language never appears in product reviews — the
  tweets fix that distribution gap. The sampled sentences live in
  `data/train_sentences.csv` (0.87 MB, committed to the repo); see
  `build_dataset.py` for how it was built.
- **Model:** TF-IDF (max 5,000 features, 1–2 grams) + Logistic Regression.
- **Test accuracy: 0.7431** (20% held-out split, 2,600 sentences). Lower than
  the old review-only 0.8083 — but the old number was measured on clean
  reviews only. On the same held-out reviews the new model still scores
  0.7997; tweets are simply noisier (0.7261 on held-out tweets), and the
  combined test set now includes them. The model did not get worse at
  reviews; the test got harder and more realistic.
- **Model file:** `model/sentiment_model.joblib` — 0.22 MB. Small on purpose, so it
  loads in under a second and fits in a serverless bundle.

## Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (reads data/train_sentences.csv — no download needed)
python train.py

# 3. Start the API
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/demo to try it in your browser — the Text sentiment
tab is the default; the Face emotion tab is at http://127.0.0.1:8000/demo#face
(the face models load only when that tab is first opened).
Or open http://127.0.0.1:8000/docs for the interactive API docs.

## Try it with curl

```bash
# Health check
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

# Positive
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this movie, it was fantastic and wonderful!"}'
# {"label":"positive","score":0.8818}

# Negative
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was terrible, a complete waste of time, I hated every minute of it."}'
# {"label":"negative","score":0.9235}

# Batch
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product, highly recommended!", "Awful experience, never again."]}'
# [{"label":"positive","score":0.8126},{"label":"negative","score":0.8485}]

# Empty text is rejected with a 400
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"text": "   "}'
# {"detail":"text must not be empty"}
```

## Run the tests

```bash
uvicorn app:app --port 8000 &
python test_api.py   # 9 checks: health, positive, negative, empty 400, missing/empty 400/422, batch, /demo, /face 307, / 307
```

## Deploy — Render

No Docker needed — Render runs it as a native Python web service:

1. Push this folder to a GitHub repo.
2. Render dashboard → **New → Web Service** → pick the repo.
3. **Environment:** Python.
4. **Build Command:** `pip install -r requirements.txt && python train.py`
5. **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Free tier works fine for this project.

The model is trained during the build (`train.py` reads the committed
`data/train_sentences.csv` — no download at build time — and saves
`model/sentiment_model.joblib`), so the trained model file itself is not
committed to the repo — it is a build artifact, regenerated on every deploy.

**Free-tier note:** a free Render service sleeps after a period of
inactivity, so the first request after it has been idle takes a few seconds
while the service wakes up and reloads the model. After that it responds
normally.

## Deploy — Vercel

The same code also deploys to Vercel — no Dockerfile, no extra service:

1. Push this folder to a GitHub repo (same repo as for Render).
2. Vercel dashboard → **Add New → Project** → import the repo (or `vercel` from the CLI).
3. Vercel auto-detects Python, installs `requirements.txt`, and serves
   `api/index.py` — which wraps the FastAPI app with [Mangum](https://mangum.io)
   (an ASGI adapter) so Vercel's Python runtime can run it. `vercel.json`
   rewrites every path to `/api/index.py`, and its build command installs
   dependencies and trains the model
   (`pip install -r requirements.txt && python train.py`), so the model file
   is generated during the build there too.

## Render vs Vercel — the honest comparison

- **Render:** a persistent server — the model loads once at startup and stays
  loaded, so every request is fast. The simpler home for an always-on API.
- **Vercel:** serverless functions — the first request after a while idle is a
  **cold start** that reloads the model (small — 0.22 MB — so this takes about
  a second). Convenient because everything you already deploy (portfolio,
  twin, Studient) lives there in one place.

## Project structure

```
ml-model-api/
├── train.py                  # trains TF-IDF + LogisticRegression on data/train_sentences.csv
├── build_dataset.py          # builds data/train_sentences.csv (UCI reviews + Sentiment140 sample)
├── app.py                    # FastAPI app: /, /health, /predict, /predict/batch, /demo (text + face tabs); /face redirects to /demo#face
├── api/
│   └── index.py              # Vercel entrypoint: FastAPI app wrapped with Mangum
├── vercel.json               # routes all paths to /api/index.py on Vercel
├── requirements.txt          # fastapi, uvicorn, scikit-learn, joblib, pydantic, mangum
├── test_api.py               # smoke tests against a running server
├── README.md
├── ML_Model_API_Build_Report.pdf  # build report (local copy; not committed)
├── model/
│   └── sentiment_model.joblib  # trained model (created by train.py, 0.22 MB; built at deploy time, not committed)
├── data/
│   ├── train_sentences.csv   # 13,000 labelled sentences (UCI + Sentiment140) — committed, read by train.py
│   └── sentiment_labelled_sentences.zip  # UCI dataset (used by build_dataset.py; not committed)
└── shots/
    ├── merged-desktop.png    # merged /demo (text tab) verified at 1280px
    ├── merged-face-desktop.png  # merged /demo (face tab) verified at 1280px
    └── merged-mobile.png     # merged /demo verified at 390px (local only)
```

## Why no Docker / no PyTorch?

- **Docker:** Docker Desktop on Windows needs installation and eats RAM.
  Nothing here needs a container — `uvicorn` runs it locally, Render runs
  native Python, and Vercel runs serverless Python.
- **PyTorch:** far too heavy for a small text classifier. TF-IDF +
  Logistic Regression reaches 0.74 accuracy on the broader dataset
  (reviews + tweets), trains in seconds, and the saved model is 0.22 MB.
