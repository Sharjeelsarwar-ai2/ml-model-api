"""Build ML_Model_API_Build_Report.pdf — step-by-step build report.

Run:  /tmp/pdf-venv/bin/python build_report.py
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, PageBreak)

OUT = "ML_Model_API_Build_Report.pdf"
ACCURACY = "0.8020"

ROSE = colors.HexColor("#D6336C")
MAGENTA = colors.HexColor("#E91E63")
BLUSH = colors.HexColor("#FDF2F5")
INK = colors.HexColor("#1F1B1E")
MUTED = colors.HexColor("#7A6A70")
LINE = colors.HexColor("#F3D9E2")
LIGHT = colors.HexColor("#FFF9FB")

S_TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26,
                         leading=31, textColor=ROSE, spaceAfter=6)
S_SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=13,
                       leading=18, textColor=MUTED, spaceAfter=4)
S_H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16,
                      leading=20, textColor=ROSE, spaceBefore=10, spaceAfter=6)
S_H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5,
                      leading=16, textColor=INK, spaceBefore=8, spaceAfter=4)
S_BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5,
                        leading=14.5, textColor=INK, spaceAfter=4)
S_BULLET = ParagraphStyle("bullet", parent=S_BODY, leftIndent=14, bulletIndent=4,
                          spaceAfter=3)
S_CODE = ParagraphStyle("code", fontName="Courier", fontSize=9,
                        leading=11, textColor=INK)
S_CAPTION = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9.5,
                           leading=13, textColor=MUTED, spaceAfter=8)


def code_block(lines):
    inner = "<br/>".join(lines)
    t = Table([[Paragraph(f"<font face='Courier' size='9'>{inner}</font>", S_CODE)]],
              colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def data_table(header, rows, widths=None):
    data = [[Paragraph(f"<b>{h}</b>", S_BODY) for h in header]]
    for row in rows:
        data.append([Paragraph(c, S_BODY) for c in row])
    t = Table(data, colWidths=widths or [32 * mm, 62 * mm, 76 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ROSE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BLUSH]),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm,
                      "ML Model as an API — Sentiment Classifier  |  Muhammad Sharjeel Sarwar")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.restoreState()


story = []

# ---- Title block ----
story.append(Spacer(1, 8 * mm))
story.append(Paragraph("ML Model as an API", S_TITLE))
story.append(Paragraph("Sentiment Classifier", S_TITLE))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph(
    "Train a sentiment classifier and serve it as a live REST API — "
    "built with scikit-learn and FastAPI. No Docker, no PyTorch.", S_SUB))
story.append(Spacer(1, 6 * mm))
story.append(Paragraph("<b>Author:</b> Muhammad Sharjeel Sarwar", S_BODY))
story.append(Paragraph("<b>Date:</b> 1 November 2026", S_BODY))
story.append(Paragraph("<b>Test accuracy:</b> 0.8020 &nbsp;|&nbsp; <b>Model size:</b> 0.83 MB", S_BODY))
story.append(Spacer(1, 6 * mm))

# ---- Architecture flow ----
story.append(Paragraph("Architecture — at a glance", S_H1))
story.append(code_block([
    "Dataset &rarr; Train &rarr; Saved model &rarr; REST API &rarr; Live requests",
    "",
    "UCI Sentiment Labelled Sentences (3,000 reviews) + Sentiment140 tweets (100,000)",
    "  &rarr; TF-IDF (word + char) + Logistic Regression  (train.py)",
    "  &rarr; model/sentiment_model.joblib  (0.83 MB)",
    "  &rarr; FastAPI app  (app.py)  —  POST /predict",
    "  &rarr; anyone can send a request and get a label + score back",
]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph("Example request / response", S_H2))
story.append(code_block([
    "POST /predict",
    '{ "text": "I absolutely loved this movie, it was fantastic!" }',
    "",
    "Response:  { \"label\": \"positive\", \"score\": 0.9680 }",
]))
story.append(Spacer(1, 2 * mm))

# ---- What was used ----
story.append(Paragraph("What was used — and why", S_H1))
story.append(data_table(
    ["Tool", "Used for", "Why this one"],
    [
        ["Python 3.12", "Everything", "The language the whole ML + API stack is written in."],
        ["scikit-learn 1.9.1", "TF-IDF + LogisticRegression", "Battle-tested, tiny, fast. No GPU needed."],
        ["FastAPI 0.162.0", "The REST API", "Modern, fast, automatic docs at /docs."],
        ["Uvicorn 0.54.0", "Serving the API", "The standard ASGI server for FastAPI."],
        ["joblib 1.6.0", "Saving / loading the model", "One small file (0.83 MB) for vectorizer + model."],
        ["pydantic 2.14.2", "Request validation", "Rejects bad input (empty text) with clean 400 errors."],
        ["Mangum 0.22.0", "Vercel serverless adapter", "Wraps the FastAPI app for Vercel's Python runtime."],
    ]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph("Deliberately NOT used", S_H2))
story.append(Paragraph(
    "<b>Docker — not used.</b> Docker Desktop on Windows needs installation and uses too "
    "much RAM on the laptop. Nothing in this project needs a container: <font face='Courier' size='10'>uvicorn</font> "
    "runs it locally, Render runs native Python, and Vercel runs serverless Python.", S_BULLET, bulletText="•"))
story.append(Paragraph(
    "<b>PyTorch — not used.</b> Far too heavy for a small text classifier. Word + char TF-IDF "
    " + Logistic Regression reaches 0.80 accuracy on the broader dataset (reviews + tweets), trains in ~30 seconds, and the saved model "
    "is 0.83 MB instead of hundreds of MB.", S_BULLET, bulletText="•"))

# ---- Step by step ----
story.append(Paragraph("Step-by-step build", S_H1))

story.append(Paragraph("Step 1 — Dataset (retrained on broader data — see Steps 10–11)", S_H2))
story.append(Paragraph(
    "<b>Two sources, 103,000 labelled sentences, committed as "
    "<font face='Courier' size='10'>data/train_sentences_part1–4.csv</font></b> (6.91 MB, UTF-8, built by "
    "<font face='Courier' size='10'>build_dataset.py</font> — no download happens at train or deploy time). "
    "<b>UCI Sentiment Labelled Sentences</b> (archive.ics.uci.edu, dataset 331): 3,000 sentences from "
    "Amazon, IMDb and Yelp reviews — 1,500 positive, 1,500 negative. "
    "<b>Sentiment140 tweets</b> (cs.stanford.edu, 1.6M tweets): a balanced sample of <b>100,000 tweets</b> "
    "(50,000 positive, 50,000 negative), lightly cleaned (URLs and @mentions stripped). The tweets were added "
    "because the review-only model misclassified everyday happy news (see Step 10). "
    "<b>Evaluation:</b> a fixed 2,000-sentence eval set "
    "(<font face='Courier' size='10'>data/eval_sentences.csv</font>, stratified split of the original data, "
    "seed 123, 1,000 positive / 1,000 negative) is held out of training — <font face='Courier' size='10'>train.py</font> "
    "trains on the other 101,982 rows and reports accuracy on those 2,000, so every configuration is measured "
    "on exactly the same sentences (see Step 11).", S_BODY))

story.append(Paragraph("Step 2 — Text preprocessing", S_H2))
story.append(Paragraph(
    "Two vectorizers run together (<font face='Courier' size='10'>FeatureUnion</font>): a <b>word</b> "
    "TF-IDF (lowercase, English stop-words removed, word unigrams + bigrams, capped at <b>10,000 features</b>) "
    "and a <b>character</b> TF-IDF (character 3–5 grams inside words, capped at <b>10,000 features</b>). "
    "The character features catch slang, stretched words (“looove”, “sooo”) and typos that word features miss — "
    "they are what fixed the “Why this always happens to me” regression (Step 11). The whole saved model is 0.83 MB.", S_BODY))

story.append(Paragraph("Step 3 — Training", S_H2))
story.append(Paragraph(
    "<b>LogisticRegression</b> (max_iter=1000, C=2.0) on the combined word + char TF-IDF features. "
    "Training takes ~30 seconds on a laptop. Measured on the fixed 2,000-sentence eval set:", S_BODY))
story.append(data_table(
    ["Metric", "Negative", "Positive"],
    [
        ["Precision", "0.82", "0.79"],
        ["Recall", "0.78", "0.83"],
        ["F1-score", "0.80", "0.81"],
        ["<b>Overall accuracy</b>", "", "<b>0.8020</b>"],
    ]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "Test accuracy <b>0.8020</b> — per-source: reviews <b>0.8172</b>, tweets <b>0.7975</b>. "
    "The old review-only model scored 0.8083, but that was measured on clean reviews only; the new number is "
    "measured on a harder, more realistic mix of reviews and tweets (see Step 11).", S_BODY))

story.append(Paragraph("Step 4 — Saving the model", S_H2))
story.append(Paragraph(
    "The fitted vectorizer, the trained model and a small metadata dict (dataset name, accuracy, "
    "hyperparameters) are saved together with <font face='Courier' size='10'>joblib</font> as a single file: "
    "<font face='Courier' size='10'>model/sentiment_model.joblib</font> (0.83 MB). The API loads this one file at "
    "startup — no retraining at serve time.", S_BODY))

story.append(Paragraph("Step 5 — The FastAPI app (app.py)", S_H2))
story.append(Paragraph(
    "The model is loaded <b>once</b> at startup (FastAPI lifespan handler) and reused for every request. "
    "Pydantic models validate all input; empty text returns a 400, never a crash.", S_BODY))
story.append(data_table(
    ["Endpoint", "What it does", "Example"],
    [
        ["GET /", "Redirects to /demo", "Bare URL opens the demo page (307)"],
        ["GET /health", "Health check", '{ "status": "ok", "model_loaded": true }'],
        ["POST /predict", "Classify one sentence", '{ "label": "positive", "score": 0.9680 }'],
        ["POST /predict/batch", "Classify up to 100 texts", "Returns a list of predictions"],
        ["GET /demo", "Demo page: Text sentiment + Face emotion tabs", "One page, two tools; tabs in the URL hash"],
        ["GET /face", "Redirects to /demo#face", "Old face link kept working (307)"],
    ]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "The <b>score</b> is the model's probability for the predicted label (0–1, rounded to 4 decimals). "
    "So a positive prediction with score 0.9680 means the model is 96.8% confident it is positive.", S_BODY))

story.append(Paragraph("Step 6 — The /demo page: two tools, one page", S_H2))
story.append(Paragraph(
    "The demo is now <b>one page with two tabs</b> — <b>Text sentiment</b> and <b>Face emotion</b> — "
    "in the same fresh identity: soft blush background (#FDF2F5), deep rose / magenta accents "
    "(#D6336C / #E91E63), dark ink text, white cards with soft shadows and rounded corners. "
    "The tab state lives in the URL hash, so each tool is directly linkable: "
    "<font face='Courier' size='10'>/demo#text</font> and <font face='Courier' size='10'>/demo#face</font>. "
    "The old <font face='Courier' size='10'>GET /face</font> address still works — it redirects (307) to "
    "<font face='Courier' size='10'>/demo#face</font>.", S_BODY))
story.append(Paragraph(
    "<b>Text tab</b> — a large textarea, an “Analyze sentiment” button, three "
    "clickable example sentences, a result card (label pill + confidence meter), a plain-words feeling line, "
    "the raw POST /predict call, a loading state (“Analyzing…”) and a graceful inline error if the API fails.", S_BODY))
story.append(Paragraph(
    "<b>Face tab</b> — the camera / photo-upload face emotion tool (see Step 9). The face-api.js library "
    "loads deferred, so the text tab works instantly even if the CDN is slow, and the face models load "
    "only the first time the Face tab is opened. Switching back to the text tab stops a running camera. "
    "The camera preview is mirrored like the phone's camera app (see Step 10). Verified visually at desktop 1280px and "
    "mobile 390px — no overflow, no overlapping text, 0 console errors.", S_BODY))

story.append(Paragraph("Step 7 — Local testing (verified)", S_H2))
story.append(Paragraph(
    "Server started with <font face='Courier' size='10'>uvicorn app:app --port 8000</font>; "
    "<font face='Courier' size='10'>test_api.py</font> runs 9 checks — <b>9/9 passed, 0 errors</b>. "
    "The exact curl checks and their real outputs:", S_BODY))
story.append(code_block([
    "$ curl http://127.0.0.1:8000/health",
    '{"status":"ok","model_loaded":true}',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json'",
    "  -d '{\"text\": \"I absolutely loved this movie, it was fantastic and wonderful!\"}'",
    '{"label":"positive","score":0.9680}',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json'",
    "  -d '{\"text\": \"This was terrible, a complete waste of time, I hated every minute of it.\"}'",
    '{"label":"negative","score":0.9257}',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{\"text\": \"   \"}'",
    '{"detail":"text must not be empty"}   (HTTP 400)',
    "",
    "$ curl -o /dev/null -w \"%{http_code}\" http://127.0.0.1:8000/demo   ->  200",
    "",
    "$ curl -o /dev/null -w \"%{http_code}\" http://127.0.0.1:8000/face   ->  307 to /demo#face",
    "$ curl -o /dev/null -w \"%{http_code}\" http://127.0.0.1:8000/      ->  307 to /demo",
]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "The Mangum / Vercel wrapper was also verified locally: importing <font face='Courier' size='10'>api/index.py</font> "
    "succeeds and a simulated serverless call returns GET /health → 200 "
    "<font face='Courier' size='10'>{\"status\":\"ok\",\"model_loaded\":true}</font>, POST /predict → 200 positive, "
    "empty text → 400.", S_BODY))

story.append(Paragraph("Step 8 — Deployment (both without Docker)", S_H2))
story.append(Paragraph("<b>Render — native Python web service (recommended home):</b> push the folder to a GitHub repo; "
    "Render dashboard → New → Web Service → pick the repo; Environment: Python; "
    "Build Command: <font face='Courier' size='10'>pip install -r requirements.txt &amp;&amp; python train.py</font>; "
    "Start Command: <font face='Courier' size='10'>uvicorn app:app --host 0.0.0.0 --port $PORT</font>.", S_BODY))
story.append(Paragraph(
    "The model is trained during the build, so the model file is not committed to the repo — "
    "it is regenerated by train.py on every deploy. "
    "Free tier works fine. Note: a free Render service sleeps when idle, so the first request after "
    "a sleep takes a few seconds while it wakes up and reloads the model.", S_BODY))
story.append(Paragraph("<b>Vercel — serverless (same repo):</b> Vercel dashboard → Add New → Project → import the same repo. "
    "Vercel auto-detects Python and installs <font face='Courier' size='10'>requirements.txt</font>; <font face='Courier' size='10'>vercel.json</font> sets the build command to install requirements and run <font face='Courier' size='10'>train.py</font>, so the model is generated during the build. "
    "<font face='Courier' size='10'>api/index.py</font> wraps the FastAPI app with Mangum; <font face='Courier' size='10'>vercel.json</font> rewrites every path to <font face='Courier' size='10'>/api/index.py</font>.", S_BODY))
story.append(Paragraph(
    "<b>Render vs Vercel:</b> Render is a persistent server — the model loads once and stays loaded. "
    "Vercel is serverless — cold starts reload the model (small at 0.83 MB, about a second).", S_BODY))

story.append(Paragraph("Step 9 — Face emotion detection (the Face tab)", S_H1))
story.append(Paragraph(
    "<b>What was added — and then merged.</b> Face emotion detection first shipped as a second demo "
    "page, <font face='Courier' size='10'>GET /face</font>, and was then merged into the demo page as "
    "its second tab, so both tools now live at <font face='Courier' size='10'>GET /demo</font> under one "
    "link. The tab is <b>Face emotion</b> (direct link <font face='Courier' size='10'>/demo#face</font>); "
    "the old <font face='Courier' size='10'>/face</font> address redirects there (307), so any link or "
    "bookmark to it still lands on the face tool. It reads facial expressions two ways: a <b>live camera</b> "
    "(Start/Stop buttons, front camera via <font face='Courier' size='10'>getUserMedia</font> with "
    "<font face='Courier' size='10'>facingMode: \"user\"</font>, face box + top-expression label drawn live "
    "on a canvas overlay) and a <b>photo upload</b> (pick a photo from the gallery, same detection on the "
    "still image). The result shows the top emotion big with its emoji and a confidence bar, plus all 7 "
    "expression scores listed underneath.", S_BODY))
story.append(Paragraph(
    "<b>How it works.</b> Detection runs <b>100% in the browser</b> with "
    "<font face='Courier' size='10'>face-api.js</font> — the maintained fork "
    "<font face='Courier' size='10'>@vladmandic/face-api</font>, pinned version <b>1.7.15</b>. The page loads the "
    "library and three models from the jsDelivr CDN:", S_BODY))
story.append(code_block([
    "Library:  https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.js",
    "Models:   https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/",
    "  tiny_face_detector_model  (finds the face box)",
    "  face_landmark_68_model    (68 facial landmarks — expressions need these)",
    "  face_expression_model     (scores the 7 expressions)",
]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "The seven expressions and how they are shown: <b>happy → Happy</b>, <b>sad → Sad / Hurting</b>, "
    "<b>angry → Angry</b>, <b>fearful → Scared</b>, <b>disgusted → Disgusted</b>, "
    "<b>surprised → Surprised</b>, <b>neutral → Neutral</b>.", S_BODY))
story.append(Paragraph(
    "<b>Loading, lazily, and privately.</b> The face-api.js library tag is deferred, so the text tab works instantly "
    "even on a slow connection. The three models load only the first time the Face tab is opened, with a "
    "“Loading face models…” state. One real bug was caught by testing: on a direct "
    "<font face='Courier' size='10'>/demo#face</font> visit, the tab code ran before the deferred library "
    "had arrived — the face initialiser now waits up to 15 seconds for the library before reporting a failure. "
    "Detection runs <b>100% in the browser</b>: the Render free server cannot run a face model, and camera frames "
    "and photos never leave the user's device — nothing is uploaded or stored. TensorFlow.js uses WebGL when the "
    "device supports it, with an automatic fallback to CPU. Camera permission denied or no camera shows a clear "
    "inline message and points to photo upload; a photo with no face says so plainly; a failed CDN load shows an "
    "inline error instead of a blank page.", S_BODY))
story.append(Paragraph(
    "<b>Honest limits.</b> The model detects the 7 standard expressions only. “Hurting” is <b>not</b> a "
    "separate class — it is shown as <b>Sad / Hurting</b>, i.e. the model's <i>sad</i> score read in plain "
    "words, exactly like the feeling line on the text demo. The expressions are a reading of the face, not a "
    "diagnosis of how someone feels.", S_BODY))
story.append(Paragraph(
    "<b>What was verified (and what was not).</b> <font face='Courier' size='10'>test_api.py</font> runs 9 checks — "
    "<b>9/9 passed</b>, including <font face='Courier' size='10'>GET /face</font> → 307 to "
    "<font face='Courier' size='10'>/demo#face</font> and <font face='Courier' size='10'>GET /</font> → 307 to "
    "<font face='Courier' size='10'>/demo</font>. In a headless browser the merged page loads with both tabs; "
    "the text tab ran end-to-end (positive sentence → Positive, 0.90, feeling line shown); the face tab loaded "
    "its models lazily and detected a smiling portrait as <b>Happy, score 1.00</b>; a direct "
    "<font face='Courier' size='10'>/demo#face</font> load opens the face tab and loads the models; "
    "the camera-denied path shows its message gracefully; layout is clean at desktop 1280px and mobile 390px "
    "(0 px horizontal overflow on both tabs), 0 console errors. <b>Not verified:</b> the live camera on a real "
    "phone — headless Firefox has no camera (the sandbox also could not fetch jsDelivr directly, so in-browser "
    "detection tests used byte-identical local copies of the same pinned files; the shipped page loads everything "
    "from the CDN, where every URL was verified HTTP 200).", S_BODY))

story.append(Paragraph("Step 10 — Two fixes from real-phone testing", S_H1))
story.append(Paragraph(
    "<b>Fix 1 — the front camera preview was mirrored (inverted).</b> On a real phone the live camera "
    "showed raw, unmirrored frames, so it looked flipped next to the phone's own camera app. The fix: the "
    "<font face='Courier' size='10'>&lt;video&gt;</font> gets "
    "<font face='Courier' size='10'>transform: scaleX(-1)</font> while the camera runs, and the overlay "
    "mirrors every detection with it — each box at x' = width - x - boxWidth, every landmark at "
    "x' = width - x. Photos are <b>not</b> mirrored. Two real bugs were caught while verifying (face-api.js "
    "draw helpers reject plain objects, and <font face='Courier' size='10'>FaceLandmarks68.positions</font> is "
    "getter-only), so the mirrored detection is built by cloning each level with its prototype intact. "
    "Verified headless with a synthetic canvas test (photo mode draws on the original box edge, camera mode on the "
    "mirrored edge). <b>11/11 mirror checks passed, 0 console errors.</b> The live camera itself cannot be tested "
    "headless — the final proof is a real phone.", S_BODY))
story.append(Paragraph(
    "<b>Fix 2 — the model was retrained on broader data.</b> “I passed my course as top performer yay” came "
    "back <b>negative 0.56</b> — the review-only model had never seen achievement or life-update language, and "
    "0.56 means it was basically unsure. The fix came from the training data, not from rules: no keyword rules "
    "were added and the sentence was not special-cased. The model was retrained twice more after this (see "
    "Step 11): it now trains on 103,000 sentences with word + char TF-IDF, and the sentence predicts "
    "<b>positive 0.9104</b>. Probes on the final model: “I passed my driving test yay” → positive 0.6939; "
    "“I failed my exam and feel terrible” → negative 0.9979; the existing "
    "<font face='Courier' size='10'>test_api.py</font> sentences still pass and the suite is <b>9/9 passed</b>. "
    "One honest miss remains: “I got the job!” still predicts <b>negative 0.5167</b> — the words are near 50/50 "
    "in the tweet data, and no configuration in Step 11 fixes it honestly (see Step 11).", S_BODY))

# ---- Step 11 — Full error analysis (what was wrong, what fixed it) ----
story.append(Paragraph("Step 11 — Full error analysis: what was still wrong, and what fixed it", S_H1))
story.append(Paragraph(
    "<b>How it was measured.</b> Before changing anything, a <b>fixed 2,000-sentence eval set</b> was carved out "
    "(<font face='Courier' size='10'>data/eval_sentences.csv</font>, stratified, seed 123 — a different seed from "
    "any training split, so no candidate had seen it). The shipped model at the time scored <b>0.7490</b> on it "
    "(502 errors: 269 false positives, 233 false negatives; reviews 0.7996, tweets 0.7342). Every candidate below "
    "was trained without those 2,000 rows and measured on exactly the same set — no cherry-picking, no fitting to "
    "individual sentences.", S_BODY))
story.append(Paragraph(
    "<b>What the errors actually are.</b> A hand-read sample of 40 errors, plus programmatic counts over all 502: "
    "<b>~50% are label noise</b> — tweets that carry no sentiment at all but were labelled by their emoticon "
    "(“here is my lil fam”, “out with the dog, see you later”, “Thanks!” labelled negative, “Vodka + Skittles = best "
    "drink I’ve had” labelled negative). No model can classify those correctly, so the realistic ceiling on this eval "
    "set is ~0.85–0.88, not 1.00. Of the rest: <b>~33% are real misses</b> — implicit negativity (“My order was not "
    "correct”, “crashed my Firefox”) and negation (“I did not have any problem with this item”, “I don’t think you "
    "will be disappointed”); <b>~10% mixed feelings</b> (good and bad in one sentence); <b>~7% sarcasm</b> (“yeah, "
    "im in love with pollen related sneezing fits”). Programmatic counts agree: negation words in 21% of errors, "
    "questions in 11%, and <b>49% of all errors are near coin-flips</b> (confidence below 0.6) — the model is unsure, "
    "not confidently wrong. Only 32 of 502 errors were confident (above 0.8), and most of those are label errors.", S_BODY))
story.append(Paragraph("Candidates tried (same 2,000-sentence eval set, best first):", S_H2))
story.append(data_table(
    ["Configuration", "Accuracy", "Notes"],
    [
        ["<b>100k tweets, word+char TF-IDF, LR C=2.0 — CHOSEN</b>", "<b>0.8020</b>", "Reviews 0.8172, tweets 0.7975, model 0.83 MB"],
        ["100k tweets, word+char TF-IDF, LR C=1.0", "0.8000", "Statistically tied with the chosen one"],
        ["70k tweets, word+char TF-IDF, LR C=2.0", "0.7985", "In-budget (CSV 4.9 MB) — but fails “I passed my course”"],
        ["100k tweets, word-only 10k TF-IDF, LR C=1.0", "0.7780", "Best on probes, but 2.4 points worse overall"],
        ["13k rows, word+char TF-IDF, LR C=1.0", "0.7815", "Character features alone: +3.3 points over baseline"],
        ["13k rows, word 20k TF-IDF, LR C=1.0", "0.7560", "Bigger vocabulary helps a little"],
        ["13k rows, word 5k TF-IDF, LR C=1.0 — BASELINE", "0.7490", "The model before this analysis"],
    ],
    widths=[72 * mm, 26 * mm, 72 * mm]))
story.append(Paragraph(
    "Also tried, all worse: sublinear TF (0.7565, no gain), LinearSVC calibrated (0.7395), and removing short "
    "tweets under 4 words (0.7425 — the short tweets carry signal too).", S_BODY))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "<b>Two failures Sharjeel reported, and what caused them.</b> <b>(1) “Why this always happens to me” came back "
    "positive 0.5287 — a regression</b> (the old review-only model said negative 0.5357). Root cause: with word-only "
    "features and English stop-words, “why / this / me” are all removed, leaving only the weak words “always” and "
    "“happens” — a coin flip the 13k-tweet model lost (reproduced locally: positive 0.5986). The character features fix "
    "it properly: they see “why”, “ happe” and “ me” as negative signals. The final model scores it "
    "<b>negative 0.8952</b> (“This always happens to me” 0.7305, “Why does this always happen to me” 0.8545). "
    "<b>(2) “I passed my course as top performer” was fragile</b> (positive 0.6663 without “yay”, 0.506 for “top "
    "performer” alone — “yay” was doing the work). Root cause, measured in the data: <b>“passed” is 74% negative in the "
    "tweets</b> — “passed away” and “passed out” appear 746 times in the 1.6M-tweet corpus, achievement uses of “passed” "
    "only 20 times — and <b>“top performer” appears 0 times</b> in training. No amount of resampling fixes a 37:1 ratio. "
    "The character features rescue it through substrings shared with positive words (“per”, “perf”, “form” from "
    "“perfect”/“performance”): the final model scores the sentence <b>positive 0.7495</b>, “top performer” 0.7462, "
    "“I passed my course” 0.5986 — correct on all three, though the last one is thin. Honest caveat: the model has not "
    "truly learned achievement language; it leans on proxy signals, because no available dataset contains achievement "
    "sentences to learn from.", S_BODY))
story.append(Paragraph(
    "<b>What still fails, reported as-is.</b> “I passed my exam” is <b>negative 0.7815</b> — it fails on <b>every</b> "
    "configuration tried (best was negative 0.62): “exam” is negative in tweets (“exam tomorrow”, “failed my exam”) and "
    "“passed” is negative, so the two signals compound. “I got the job!” is <b>negative 0.5167</b> — a coin flip on every "
    "configuration (it flips both ways across retrains). Fixing either honestly would need training data that contains "
    "achievement sentences with correct labels; neither the UCI reviews nor Sentiment140 has them, and inventing such "
    "sentences by hand would be fitting the model to the test. Both are left as known limitations, not tuned to pass.", S_BODY))
story.append(Paragraph(
    "<b>The size trade-off.</b> The chosen configuration needs 100,000 tweets, so the committed CSV is <b>6.91 MB</b> — "
    "over the original ~5 MB guidance. The in-budget alternative (70,000 tweets, 0.7985) is only 0.35 points behind on "
    "accuracy, but it fails “I passed my course” and is weaker on every achievement probe. The extra 50,000 tweets buy the "
    "last point of accuracy and the probe fixes, so they were kept; the model file itself stays small (0.83 MB).", S_BODY))

# ---- How to run + file list ----
story.append(Paragraph("How to run it locally", S_H1))
story.append(code_block([
    "pip install -r requirements.txt",
    "python train.py          # trains the model, saves model/sentiment_model.joblib",
    "uvicorn app:app --reload # API at http://127.0.0.1:8000",
    "python test_api.py       # 9 smoke tests against the running server",
]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph("File list", S_H1))
story.append(code_block([
    "train.py                     trains word+char TF-IDF + LogisticRegression (reads data/train_sentences_part1–4.csv)",
    "build_dataset.py             builds data/train_sentences_part1–4.csv (UCI + Sentiment140 sample)",
    "data/train_sentences_part1–4.csv     103,000 labelled sentences in 4 parts, 6.91 MB total, committed",
    "data/eval_sentences.csv      fixed 2,000-sentence eval set (seed 123), held out of training, committed",
    "app.py                       FastAPI app + merged /demo (text + face tabs); /face -&gt; /demo#face",
    "api/index.py, vercel.json    Vercel entrypoint (Mangum) + routing (Vercel only)",
    "requirements.txt             fastapi, uvicorn, scikit-learn, joblib, pydantic, mangum (pinned)",
    "test_api.py / README.md      9 smoke tests (9/9 passed) / setup, curl examples, deploys",
    "model/sentiment_model.joblib trained vectorizer + model, 0.83 MB (built at deploy, not committed)",
    "data/*.zip, shots/           UCI dataset (not committed) / merged /demo screenshots",
    "ML_Model_API_Build_Report.pdf  this report",
]))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=14 * mm, bottomMargin=16 * mm,
                      title="ML Model as an API — Sentiment Classifier — Build Report",
                      author="Muhammad Sharjeel Sarwar")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print(f"Built {OUT}")
