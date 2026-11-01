"""
app.py — ML Model as an API.

Serves the sentiment classifier trained by train.py (TF-IDF + LogisticRegression,
scikit-learn only) as a FastAPI REST API. No Docker, no PyTorch.

Endpoints:
    GET  /               — project info
    GET  /health         — {"status": "ok", "model_loaded": true}
    POST /predict        — {"text": "..."} -> {"label": "positive", "score": 0.94}
    POST /predict/batch  — {"texts": [...]} -> [{"label": ..., "score": ...}, ...]
    GET  /demo           — demo page: text sentiment + face emotion tabs
    GET  /face           — redirects to /demo#face (face emotion tab)

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
<script defer src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.js"></script>
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
  /* Face tab */
  .stage {
    position: relative; width: 100%; background: var(--plum); border-radius: 14px;
    overflow: hidden; min-height: 220px; display: flex; align-items: center; justify-content: center;
  }
  .stage video, .stage img { display: block; max-width: 100%; height: auto; max-height: 480px; }
  /* The front camera shows a mirror image (like the phone's camera app), so the
     video is flipped horizontally. The overlay canvas is NOT flipped — the
     detection boxes are mirrored in JS instead (see mirrorDet), so box and
     video stay aligned. Photos are never mirrored. */
  .stage video.mirrored { transform: scaleX(-1); }
  .media { position: relative; line-height: 0; }
  .media canvas { position: absolute; left: 0; top: 0; }
  .stage .hint { color: #F8BBD0; font-size: 14px; padding: 40px 20px; text-align: center; line-height: 1.6; }
  .btnrow { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
  .btn {
    flex: 1 1 140px; font: inherit; font-size: 15px; font-weight: 700; text-align: center;
    cursor: pointer; padding: 13px 16px; border-radius: 12px; border: none;
    transition: filter .15s, transform .15s;
  }
  .btn.primary { color: #fff; background: linear-gradient(135deg, var(--magenta), var(--rose)); box-shadow: 0 8px 20px rgba(214,51,108,.30); }
  .btn.primary:hover:not(:disabled) { filter: brightness(1.06); transform: translateY(-1px); }
  .btn.ghost { background: var(--rose-soft); color: var(--ink); border: 1px solid rgba(214,51,108,.28); }
  .btn.ghost:hover:not(:disabled) { background: #FAD9E6; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  .status { margin-top: 16px; font-size: 13px; color: var(--muted); line-height: 1.5; }
  .status b { color: var(--rose); }
  .top-emotion { font-size: 26px; font-weight: 800; letter-spacing: -0.01em; line-height: 1.25; }
  .top-emotion small { display: block; font-size: 13px; font-weight: 400; color: var(--muted); margin-top: 4px; line-height: 1.5; }
  .expr-row { display: flex; align-items: center; gap: 10px; margin-top: 9px; font-size: 14px; }
  .expr-row span { width: 132px; flex-shrink: 0; }
  .expr-row .bar { flex: 1; height: 8px; border-radius: 999px; background: #F3D9E2; overflow: hidden; }
  .expr-row .bar > div { height: 100%; background: linear-gradient(90deg, var(--magenta), var(--rose)); border-radius: 999px; }
  .expr-row b { width: 42px; text-align: right; flex-shrink: 0; }
  /* Tabs */
  [hidden] { display: none !important; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab {
    flex: 1; font: inherit; font-size: 15px; font-weight: 700; cursor: pointer;
    color: var(--muted); background: #fff; border: 1px solid var(--line);
    border-radius: 14px; padding: 13px 10px; transition: all .15s;
  }
  .tab.active {
    color: #fff; background: linear-gradient(135deg, var(--magenta), var(--rose));
    border-color: transparent; box-shadow: 0 8px 20px rgba(214,51,108,.30);
  }
  .tab:hover:not(.active) { background: var(--rose-soft); }
  @media (max-width: 520px) {
    .expr-row span { width: 108px; font-size: 13px; }
    .top-emotion { font-size: 22px; }
  }
</style></head>
<body>
  <div class="wrap">
    <span class="badge">Live demo &middot; Text sentiment + Face emotion</span>
    <h1>ML Model as an API<br><span>Sentiment Classifier</span></h1>
    <p class="sub">One page, two tools. <b style="color:var(--ink)">Text sentiment</b> predicts whether a sentence is
      <b style="color:var(--ink)">positive</b> or <b style="color:var(--ink)">negative</b> through
      <code>POST /predict</code> — trained on 3,000 real Amazon, IMDb &amp; Yelp reviews
      plus 10,000 tweets, so it also understands everyday happy news
      (“I passed my course, yay!”) — not just review language.
      <b style="color:var(--ink)">Face emotion</b> reads expressions from your camera or a photo,
      entirely in your browser.</p>

    <div class="tabs" role="tablist" aria-label="Demo tools">
      <button class="tab active" id="tab-text" role="tab" aria-selected="true" aria-controls="panel-text" type="button">&#128172; Text sentiment</button>
      <button class="tab" id="tab-face" role="tab" aria-selected="false" aria-controls="panel-face" type="button">&#128512; Face emotion</button>
    </div>

    <div class="card" id="panel-text" role="tabpanel" aria-labelledby="tab-text">
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

    <div class="card" id="panel-face" role="tabpanel" aria-labelledby="tab-face" hidden>
      <div class="stage" id="stage">
        <div class="hint" id="stageHint">Camera and photo preview appear here.<br>Start the camera or upload a photo below.</div>
        <div class="media" id="media">
          <video id="video" autoplay muted playsinline style="display:none"></video>
          <img id="photo" alt="" style="display:none">
          <canvas id="overlay" width="0" height="0"></canvas>
        </div>
      </div>
      <div class="btnrow">
        <button class="btn primary" id="startCam" type="button" disabled>&#128247; Start camera</button>
        <button class="btn ghost" id="stopCam" type="button" disabled>&#9209; Stop camera</button>
        <label class="btn ghost" for="file" style="cursor:pointer">&#128444;&#65039; Upload a photo</label>
        <input type="file" id="file" accept="image/*" hidden>
      </div>
      <p class="status" id="faceStatus">Face models load when you open this tab.</p>

      <div class="result" id="faceResult">
        <div class="result-inner">
          <div class="top-emotion" id="topEmotion">&mdash;</div>
          <div class="meter-label"><span>Confidence</span><b id="faceScore">0.00</b></div>
          <div class="meter" id="faceMeter"><div></div></div>
          <div id="allExprs" style="margin-top:14px"></div>
          <p class="raw">face-api.js (tinyFaceDetector + landmarks + expressions) &mdash; all 7 expression scores above.</p>
        </div>
      </div>
      <div class="error" id="faceError"></div>
    </div>

    <footer>Powered by <code>POST /predict</code> &middot; Model accuracy 0.74 on held-out sentences<br>
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
  
    var MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model';
    var EMOTIONS = {
      happy: 'Happy 😊', sad: 'Sad / Hurting 💔', angry: 'Angry 😠',
      fearful: 'Scared 😨', disgusted: 'Disgusted 🤢',
      surprised: 'Surprised 😲', neutral: 'Neutral 😐'
    };
    var ORDER = ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral'];

    var video = document.getElementById('video');
    var photo = document.getElementById('photo');
    var overlay = document.getElementById('overlay');
    var stageHint = document.getElementById('stageHint');
    var startBtn = document.getElementById('startCam');
    var stopBtn = document.getElementById('stopCam');
    var fileInput = document.getElementById('file');
    var faceStatus = document.getElementById('faceStatus');
    var faceResult = document.getElementById('faceResult');
    var topEmotion = document.getElementById('topEmotion');
    var faceScore = document.getElementById('faceScore');
    var faceMeter = document.getElementById('faceMeter');
    var allExprs = document.getElementById('allExprs');
    var faceErr = document.getElementById('faceError');

    var modelsReady = false, stream = null, rafId = null, detecting = false, mode = null;

    function showError(msg) { faceErr.textContent = msg; faceErr.classList.add('show'); }
    function hideError() { faceErr.classList.remove('show'); }

    function detectorOptions() {
      return new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
    }

    // Mirror a detection horizontally for the camera preview: the front
    // camera video is flipped with scaleX(-1), so a box at x must be drawn
    // at x' = width - x - boxWidth (and every landmark x' = width - x) to
    // stay on the face. Photos are never mirrored — this is only used when
    // mode === 'camera'.
    // The mirrored object must keep the ORIGINAL's shape and prototypes:
    // face-api.js's draw helpers duck-type their input, and a plain
    // {detection, landmarks} object makes drawDetections throw
    // ("Box.constructor - expected box to be IBoundingBox | IRect").
    // So we clone each level, replacing only the box / positions.
    function cloneWith(obj, patch) {
      // Object.assign cannot override a getter-only prototype property
      // (FaceLandmarks68.positions is one), so copy descriptors and define
      // the patched keys as plain data properties that shadow the getters.
      var c = Object.create(Object.getPrototypeOf(obj));
      Object.keys(obj).forEach(function (k) {
        if (!(k in patch)) {
          Object.defineProperty(c, k, Object.getOwnPropertyDescriptor(obj, k));
        }
      });
      Object.keys(patch).forEach(function (k) {
        Object.defineProperty(c, k, { value: patch[k], writable: true,
                                      enumerable: true, configurable: true });
      });
      return c;
    }

    function mirrorDet(det, width) {
      var box = det.detection.box;
      var mBox = new faceapi.Box({ x: width - box.x - box.width, y: box.y,
                                   width: box.width, height: box.height });
      var mLandmarks = cloneWith(det.landmarks, {
        positions: det.landmarks.positions.map(function (p) {
          return new faceapi.Point(width - p.x, p.y);
        })
      });
      var mDetection = cloneWith(det.detection, { box: mBox });
      return cloneWith(det, { detection: mDetection, landmarks: mLandmarks });
    }

    function drawDetections(source, det) {
      var displaySize = { width: source.clientWidth, height: source.clientHeight };
      faceapi.matchDimensions(overlay, displaySize);
      var resized = faceapi.resizeResults(det, displaySize);
      var mirrored = (mode === 'camera');
      var shown = mirrored ? mirrorDet(resized, displaySize.width) : resized;
      overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
      faceapi.draw.drawDetections(overlay, shown);
      faceapi.draw.drawFaceLandmarks(overlay, shown);
      var top = topExpression(det.expressions);
      var label = EMOTIONS[top.key] + ' ' + Math.round(top.value * 100) + '%';
      var anchor = mirrored
        ? new faceapi.Point(displaySize.width - resized.detection.box.x - resized.detection.box.width,
                            resized.detection.box.y)
        : resized.detection.box.topLeft;
      new faceapi.draw.DrawTextField([label], anchor).draw(overlay);
    }

    function topExpression(expressions) {
      var best = { key: 'neutral', value: 0 };
      ORDER.forEach(function (k) {
        if (expressions[k] > best.value) best = { key: k, value: expressions[k] };
      });
      return best;
    }

    function showResult(expressions) {
      var top = topExpression(expressions);
      topEmotion.innerHTML = EMOTIONS[top.key] +
        '<small>The model is most confident this face looks ' +
        EMOTIONS[top.key].split(' ')[0].toLowerCase() +
        ' &mdash; see all 7 expression scores below.</small>';
      faceScore.textContent = top.value.toFixed(2);
      faceMeter.firstElementChild.style.width = Math.round(top.value * 100) + '%';
      var sorted = ORDER.slice().sort(function (a, b) { return expressions[b] - expressions[a]; });
      allExprs.innerHTML = sorted.map(function (k) {
        var v = expressions[k];
        return '<div class="expr-row"><span>' + EMOTIONS[k] + '</span>' +
          '<div class="bar"><div style="width:' + Math.round(v * 100) + '%"></div></div>' +
          '<b>' + v.toFixed(2) + '</b></div>';
      }).join('');
      faceResult.classList.add('show');
    }

    async function detectOnce(source) {
      var det = await faceapi.detectSingleFace(source, detectorOptions())
        .withFaceLandmarks().withFaceExpressions();
      if (!det) return null;
      drawDetections(source, det);
      showResult(det.expressions);
      return det;
    }

    async function cameraLoop() {
      if (!stream || video.paused || video.ended) return;
      if (!detecting) {
        detecting = true;
        try {
          var det = await faceapi.detectSingleFace(video, detectorOptions())
            .withFaceLandmarks().withFaceExpressions();
          if (det) { drawDetections(video, det); showResult(det.expressions); }
        } catch (e) { /* keep looping; a single failed frame is not fatal */ }
        detecting = false;
      }
      rafId = requestAnimationFrame(cameraLoop);
    }

    async function startCamera() {
      hideError();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('This browser has no camera support — please upload a photo instead.');
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false
        });
      } catch (e) {
        if (e && e.name === 'NotAllowedError') {
          showError('Camera access was denied. Allow camera permission in the browser, or upload a photo instead.');
        } else if (e && e.name === 'NotFoundError') {
          showError('No camera was found on this device — please upload a photo instead.');
        } else {
          showError('Could not start the camera (' + (e && e.name ? e.name : 'unknown error') + ') — please upload a photo instead.');
        }
        return;
      }
      stopPhoto();
      photo.style.display = 'none';
      video.srcObject = stream;
      await video.play();
      video.style.display = 'block';
      video.classList.add('mirrored'); // front camera: mirror like the phone's camera app
      stageHint.style.display = 'none';
      mode = 'camera';
      startBtn.disabled = true;
      stopBtn.disabled = false;
      faceStatus.innerHTML = '<b>Camera live</b> — detecting expressions in real time.';
      cameraLoop();
    }

    function stopCamera() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
      if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
      video.pause();
      video.srcObject = null;
      video.style.display = 'none';
      video.classList.remove('mirrored');
      mode = null;
      startBtn.disabled = !modelsReady;
      stopBtn.disabled = true;
      overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
      faceResult.classList.remove('show');
      stageHint.style.display = 'block';
      faceStatus.innerHTML = 'Camera stopped. Start it again, or upload a photo.';
    }

    function stopPhoto() {
      photo.style.display = 'none';
      photo.removeAttribute('src');
      faceResult.classList.remove('show');
    }

    fileInput.addEventListener('change', async function () {
      hideError();
      var file = fileInput.files && fileInput.files[0];
      if (!file) return;
      if (!modelsReady) { showError('Face models are still loading — please wait a moment and try again.'); return; }
      stopCamera();
      var url = URL.createObjectURL(file);
      photo.onload = async function () {
        photo.style.display = 'block';
        stageHint.style.display = 'none';
        mode = 'photo';
        faceStatus.innerHTML = '<b>Photo loaded</b> — detecting the expression&hellip;';
        try {
          var det = await detectOnce(photo);
          if (!det) {
            showError('No face found in this photo — try a clearer, front-facing photo with the face visible.');
            faceStatus.innerHTML = 'No face found. Try another photo.';
          } else {
            faceStatus.innerHTML = '<b>Done</b> — expression detected from your photo.';
          }
        } catch (e) {
          showError('Detection failed (' + (e && e.message ? e.message : 'unknown error') + ') — try another photo.');
        }
        URL.revokeObjectURL(url);
      };
      photo.onerror = function () {
        showError('That file could not be read as an image — please pick a JPG or PNG photo.');
        URL.revokeObjectURL(url);
      };
      photo.src = url;
    });

    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);

    async function initFace() {
      faceStatus.textContent = 'Loading face models\u2026';
      // The face-api.js library loads with defer, so on a direct /demo#face
      // visit it may not be here yet when this first runs — wait for it.
      var waited = 0;
      while (typeof faceapi === 'undefined' && waited < 15000) {
        await new Promise(function (r) { setTimeout(r, 200); });
        waited += 200;
      }
      if (typeof faceapi === 'undefined') {
        faceStatus.textContent = 'Could not load face-api.js.';
        showError('The face detection library could not be loaded from the CDN — check your internet connection and reload the page.');
        return;
      }
      try {
        try {
          await faceapi.tf.setBackend('webgl');
          await faceapi.tf.ready();
        } catch (e) {
          await faceapi.tf.setBackend('cpu');
          await faceapi.tf.ready();
        }
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
        ]);
        modelsReady = true;
        faceStatus.innerHTML = '<b>Models ready</b> — start the camera or upload a photo.';
        startBtn.disabled = false;
      } catch (e) {
        faceStatus.textContent = 'Face models could not be loaded.';
        showError('The face models could not be loaded from the CDN (' + (e && e.message ? e.message : 'unknown error') + ') — check your connection and reload the page.');
      }
    }
  
    // ---- Tabs: Text sentiment / Face emotion (state in the URL hash) ----
    var tabTextBtn = document.getElementById('tab-text');
    var tabFaceBtn = document.getElementById('tab-face');
    var panelText = document.getElementById('panel-text');
    var panelFace = document.getElementById('panel-face');
    var faceInitStarted = false;

    function openTab(name) {
      var isFace = (name === 'face');
      tabTextBtn.classList.toggle('active', !isFace);
      tabFaceBtn.classList.toggle('active', isFace);
      tabTextBtn.setAttribute('aria-selected', String(!isFace));
      tabFaceBtn.setAttribute('aria-selected', String(isFace));
      panelText.hidden = isFace;
      panelFace.hidden = !isFace;
      if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
      if (isFace) {
        if (!faceInitStarted) { faceInitStarted = true; initFace(); }
      } else if (typeof mode !== 'undefined' && mode === 'camera') {
        stopCamera();
      }
    }
    tabTextBtn.addEventListener('click', function () { openTab('text'); });
    tabFaceBtn.addEventListener('click', function () { openTab('face'); });
    window.addEventListener('hashchange', function () {
      openTab(location.hash === '#face' ? 'face' : 'text');
    });
    openTab(location.hash === '#face' ? 'face' : 'text');
  </script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return DEMO_HTML




@app.get("/face")
def face():
    # Old /face link kept working — the face tool now lives on /demo#face.
    return RedirectResponse(url="/demo#face", status_code=307)
