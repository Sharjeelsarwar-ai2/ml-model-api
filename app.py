"""
app.py — ML Model as an API.

Serves the sentiment classifier trained by train.py (TF-IDF + LogisticRegression,
scikit-learn only) as a FastAPI REST API. No Docker, no PyTorch.

Endpoints:
    GET  /               — project info
    GET  /health         — {"status": "ok", "model_loaded": true}
    POST /predict        — {"text": "..."} -> {"label": "positive", "score": 0.94}
    POST /predict/batch  — {"texts": [...]} -> [{"label": ..., "score": ...}, ...]
    GET  /demo           — tiny in-browser demo page

Run locally:
    uvicorn app:app --reload
"""

import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "sentiment_model.joblib")

# Loaded once at startup.
ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python train.py` first."
        )
    ARTIFACTS.update(joblib.load(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH} "
          f"(test accuracy: {ARTIFACTS['meta']['test_accuracy']})")
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="ML Model as an API — Sentiment Classifier",
    description="TF-IDF + Logistic Regression sentiment classifier served as a REST API.",
    version="1.0.0",
    lifespan=lifespan,
)

LABELS = ["negative", "positive"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100,
                             description="List of texts to classify (max 100)")


class Prediction(BaseModel):
    label: str
    score: float


def predict_one(text: str) -> Prediction:
    vectorizer = ARTIFACTS["vectorizer"]
    model = ARTIFACTS["model"]
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    idx = int(probs.argmax())
    return Prediction(label=LABELS[idx], score=round(float(probs[idx]), 4))


@app.get("/")
def root():
    # Opening the bare URL in a browser should show the demo page,
    # not a wall of JSON — API clients can use /health, /predict and /docs.
    return RedirectResponse(url="/demo", status_code=307)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ARTIFACTS}


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return predict_one(req.text)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchPredictRequest):
    if any(not t.strip() for t in req.texts):
        raise HTTPException(status_code=400, detail="texts must not contain empty strings")
    return [predict_one(t) for t in req.texts]


DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment API — Live Demo</title>
<style>
  :root {
    --rose: #D6336C; --magenta: #E91E63; --blush: #FDF2F5; --ink: #1F1B1E;
    --card: #FFFFFF; --muted: #7A6A70; --line: #F3D9E2; --plum: #2B1A20;
    --coral: #C2410C; --rose-soft: #FCE7F0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--blush); color: var(--ink); min-height: 100vh;
    background-image: radial-gradient(720px 320px at 85% -5%, rgba(233,30,99,.10), transparent 60%),
                      radial-gradient(640px 300px at 8% 0%, rgba(214,51,108,.08), transparent 60%);
  }
  .wrap { max-width: 680px; margin: 0 auto; padding: 56px 20px 64px; }
  .badge {
    display: inline-block; font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--rose); border: 1px solid rgba(214,51,108,.45); border-radius: 999px;
    padding: 6px 12px; margin-bottom: 18px; background: #fff;
  }
  h1 { font-size: 34px; line-height: 1.15; letter-spacing: -0.02em; margin-bottom: 12px; }
  h1 span { color: var(--rose); }
  .sub { color: var(--muted); font-size: 16px; line-height: 1.6; margin-bottom: 28px; }
  .sub code { color: var(--magenta); background: rgba(233,30,99,.08); padding: 1px 6px; border-radius: 6px; font-size: 14px; }
  .card {
    background: var(--card);
    border: 1px solid var(--line); border-radius: 20px; padding: 24px;
    box-shadow: 0 18px 44px rgba(214,51,108,.10), 0 2px 8px rgba(31,27,30,.04);
  }
  label { display: block; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
  textarea {
    width: 100%; min-height: 110px; resize: vertical; font: inherit; font-size: 16px; line-height: 1.55;
    color: var(--ink); background: #FFF9FB; border: 1px solid var(--line); border-radius: 14px;
    padding: 14px 16px; outline: none; transition: border-color .15s, box-shadow .15s;
  }
  textarea:focus { border-color: var(--rose); box-shadow: 0 0 0 3px rgba(214,51,108,.16); }
  textarea::placeholder { color: #B49AA3; }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
  .examples span { font-size: 13px; color: var(--muted); align-self: center; margin-right: 2px; }
  .chip {
    font: inherit; font-size: 13px; color: var(--ink); cursor: pointer;
    background: var(--rose-soft); border: 1px solid rgba(214,51,108,.28);
    padding: 7px 12px; border-radius: 999px; transition: background .15s, transform .15s;
  }
  .chip:hover { background: #FAD9E6; transform: translateY(-1px); }
  .chip.teal { background: #F3E8EE; border-color: rgba(122,31,61,.3); }
  .chip.teal:hover { background: #EAD9E0; }
  .chip.coral { background: #FDF0E7; border-color: rgba(194,65,12,.3); }
  .chip.coral:hover { background: #FBE3D3; }
  button#go {
    margin-top: 20px; width: 100%; font: inherit; font-size: 16px; font-weight: 700;
    color: #fff; background: linear-gradient(135deg, var(--magenta), var(--rose)); border: none; border-radius: 14px;
    padding: 15px 20px; cursor: pointer; transition: filter .15s, transform .15s;
    box-shadow: 0 10px 24px rgba(214,51,108,.30);
  }
  button#go:hover:not(:disabled) { filter: brightness(1.06); transform: translateY(-1px); }
  button#go:disabled { opacity: .65; cursor: wait; }
  .result { margin-top: 20px; display: none; }
  .result.show { display: block; animation: rise .3s ease; }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .result-inner {
    background: #FFF9FB; border: 1px solid var(--line); border-radius: 16px; padding: 20px;
  }
  .pill {
    display: inline-block; font-size: 15px; font-weight: 700; letter-spacing: .04em;
    padding: 8px 16px; border-radius: 999px; text-transform: capitalize;
  }
  .pill.positive { background: linear-gradient(135deg, var(--magenta), var(--rose)); color: #fff; }
  .pill.negative { background: var(--plum); color: #F8BBD0; }
  .meter-label { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); margin: 16px 0 8px; }
  .meter-label b { color: var(--ink); font-size: 15px; }
  .meter { height: 10px; border-radius: 999px; background: #F3D9E2; overflow: hidden; }
  .meter > div { height: 100%; border-radius: 999px; width: 0; transition: width .5s ease; }
  .meter.positive > div { background: linear-gradient(90deg, var(--magenta), var(--rose)); }
  .meter.negative > div { background: linear-gradient(90deg, #8A5A6B, var(--plum)); }
  .feeling { margin-top: 14px; font-size: 15px; line-height: 1.55; color: var(--ink); }
  .raw { margin-top: 14px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: var(--muted); word-break: break-word; }
  .error {
    display: none; margin-top: 20px; padding: 14px 16px; border-radius: 14px;
    background: rgba(194,65,12,.08); border: 1px solid rgba(194,65,12,.4); color: var(--coral);
    font-size: 14px; line-height: 1.5;
  }
  .error.show { display: block; }
  footer { text-align: center; color: var(--muted); font-size: 13px; margin-top: 28px; line-height: 1.6; }
  footer code { color: var(--magenta); }
  footer a { color: var(--rose); text-decoration: none; }
  @media (max-width: 520px) {
    .wrap { padding: 36px 16px 48px; }
    h1 { font-size: 27px; }
    .card { padding: 18px; }
    .sub { font-size: 15px; }
  }
</style></head>
<body>
  <div class="wrap">
    <span class="badge">Live demo &middot; TF-IDF + Logistic Regression</span>
    <h1>ML Model as an API<br><span>Sentiment Classifier</span></h1>
    <p class="sub">Type any sentence and the model predicts whether it is
      <b style="color:var(--ink)">positive</b> or <b style="color:var(--ink)">negative</b> —
      live, through <code>POST /predict</code>. Trained on 3,000 real Amazon, IMDb &amp; Yelp reviews.</p>

    <div class="card">
      <label for="text">Your sentence</label>
      <textarea id="text" placeholder="e.g. I absolutely loved this movie, it was fantastic!"></textarea>
      <div class="examples">
        <span>Try:</span>
        <button type="button" class="chip" data-t="I absolutely loved this movie, it was fantastic and wonderful!">&#128512; Clearly positive</button>
        <button type="button" class="chip coral" data-t="This was terrible, a complete waste of time, I hated every minute of it.">&#128542; Clearly negative</button>
        <button type="button" class="chip teal" data-t="The food was great but the service was slow and the place was too noisy.">&#129300; Mixed feelings</button>
      </div>
      <button id="go" type="button" onclick="run()">Analyze sentiment</button>

      <div class="result" id="result">
        <div class="result-inner">
          <span class="pill" id="pill">positive</span>
          <div class="meter-label"><span>Confidence</span><b id="score">0.94</b></div>
          <div class="meter" id="meter"><div></div></div>
          <p class="feeling" id="feeling"></p>
          <p class="raw" id="raw"></p>
        </div>
      </div>
      <div class="error" id="error"></div>
    </div>

    <footer>Powered by <code>POST /predict</code> &middot; Model accuracy 0.81 on held-out reviews<br>
      <a href="/docs">API docs</a> &middot; <a href="/health">Health check</a></footer>
  </div>
  <script>
    const box = document.getElementById('text');
    const btn = document.getElementById('go');
    const result = document.getElementById('result');
    const pill = document.getElementById('pill');
    const scoreEl = document.getElementById('score');
    const meter = document.getElementById('meter');
    const raw = document.getElementById('raw');
    const feeling = document.getElementById('feeling');
    const err = document.getElementById('error');

    // Plain-words reading of the binary result (label + confidence).
    // The model only predicts positive/negative — this line just says
    // what that means in human terms; it is not emotion detection.
    function feelingFor(label, score) {
      if (label === 'positive') {
        if (score >= 0.8) return '\\uD83D\\uDE0A Happy \\u2014 this sounds genuinely pleased and positive.';
        if (score >= 0.6) return '\\uD83D\\uDE42 Pleased \\u2014 leaning positive, in a good mood.';
        return '\\uD83E\\uDD14 A little pleased \\u2014 slightly positive, with mixed feelings.';
      }
      if (score >= 0.8) return '\\uD83D\\uDC94 Hurting \\u2014 this sounds clearly upset or unhappy.';
      if (score >= 0.6) return '\\uD83D\\uDE14 Down \\u2014 leaning negative, like something is bothering them.';
      return '\\uD83D\\uDE15 A bit down \\u2014 slightly negative, with mixed feelings.';
    }

    document.querySelectorAll('.chip').forEach(c =>
      c.addEventListener('click', () => { box.value = c.dataset.t; box.focus(); run(); }));

    async function run() {
      const text = box.value;
      err.classList.remove('show');
      if (!text.trim()) {
        err.textContent = 'Please type a sentence first — the box is empty.';
        err.classList.add('show');
        return;
      }
      btn.disabled = true;
      btn.textContent = 'Analyzing…';
      result.classList.remove('show');
      try {
        const r = await fetch('/predict', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }) });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || ('Request failed (' + r.status + ')'));
        pill.textContent = data.label;
        pill.className = 'pill ' + data.label;
        scoreEl.textContent = data.score.toFixed(2);
        meter.className = 'meter ' + data.label;
        meter.firstElementChild.style.width = Math.round(data.score * 100) + '%';
        feeling.textContent = feelingFor(data.label, data.score);
        raw.textContent = 'POST /predict \\u2192 ' + JSON.stringify(data);
        result.classList.add('show');
      } catch (e) {
        err.textContent = 'Something went wrong: ' + e.message + ' — please try again.';
        err.classList.add('show');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze sentiment';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return DEMO_HTML
