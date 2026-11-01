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
- `GET /` opens the demo page in a browser (it redirects to `/demo`)
- `GET /health`, `POST /predict/batch` — health check, batch predictions
- `GET /face` — face emotion detection in the browser: camera or photo upload,
  7 expressions (happy, sad, angry, scared, disgusted, surprised, neutral),
  100% client-side with face-api.js — nothing is sent to the server

## The model

- **Dataset:** UCI Sentiment Labelled Sentences — 3,000 labelled sentences from
  Amazon, IMDb and Yelp reviews (1,500 positive / 1,500 negative).
- **Model:** TF-IDF (max 5,000 features, 1–2 grams) + Logistic Regression.
- **Test accuracy: 0.8083** (20% held-out split, 600 sentences).
- **Model file:** `model/sentiment_model.joblib` — 0.23 MB. Small on purpose, so it
  loads in under a second and fits in a serverless bundle.

## Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (downloads the UCI dataset once, ~85 KB)
python train.py

# 3. Start the API
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/demo to try it in your browser,
or http://127.0.0.1:8000/docs for the interactive API docs.

## Try it with curl

```bash
# Health check
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

# Positive
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this movie, it was fantastic and wonderful!"}'
# {"label":"positive","score":0.896}

# Negative
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was terrible, a complete waste of time, I hated every minute of it."}'
# {"label":"negative","score":0.8478}

# Batch
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product, highly recommended!", "Awful experience, never again."]}'
# [{"label":"positive","score":0.8399},{"label":"negative","score":0.7788}]

# Empty text is rejected with a 400
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"text": "   "}'
# {"detail":"text must not be empty"}
```

## Run the tests

```bash
uvicorn app:app --port 8000 &
python test_api.py   # 8 checks: health, positive, negative, empty 400, batch, /demo, /
```

## Deploy — Render

No Docker needed — Render runs it as a native Python web service:

1. Push this folder to a GitHub repo.
2. Render dashboard → **New → Web Service** → pick the repo.
3. **Environment:** Python.
4. **Build Command:** `pip install -r requirements.txt && python train.py`
5. **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Free tier works fine for this project.

The model is trained during the build (`train.py` downloads the UCI dataset
once and saves `model/sentiment_model.joblib`), so the trained model file
itself is not committed to the repo — it is a build artifact, regenerated
on every deploy.

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
  **cold start** that reloads the model (small — 0.23 MB — so this takes about
  a second). Convenient because everything you already deploy (portfolio,
  twin, Studient) lives there in one place.

## Project structure

```
ml-model-api/
├── train.py                  # trains TF-IDF + LogisticRegression on the UCI dataset
├── app.py                    # FastAPI app: /, /health, /predict, /predict/batch, /demo, /face
├── api/
│   └── index.py              # Vercel entrypoint: FastAPI app wrapped with Mangum
├── vercel.json               # routes all paths to /api/index.py on Vercel
├── requirements.txt          # fastapi, uvicorn, scikit-learn, joblib, pydantic, mangum
├── test_api.py               # smoke tests against a running server
├── README.md
├── ML_Model_API_Build_Report.pdf  # build report (local copy; not committed)
├── model/
│   └── sentiment_model.joblib  # trained model (created by train.py, 0.23 MB; built at deploy time, not committed)
├── data/
│   └── sentiment_labelled_sentences.zip  # UCI dataset (downloaded by train.py; not committed)
└── shots/
    ├── demo-desktop.png      # /demo verified at 1280px (local only)
    └── demo-mobile.png       # /demo verified at 390px (local only)
```

## Why no Docker / no PyTorch?

- **Docker:** Docker Desktop on Windows needs installation and eats RAM.
  Nothing here needs a container — `uvicorn` runs it locally, Render runs
  native Python, and Vercel runs serverless Python.
- **PyTorch:** far too heavy for a small text classifier. TF-IDF +
  Logistic Regression reaches 0.81 accuracy on this dataset, trains in
  seconds, and the saved model is 0.23 MB.
