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
ACCURACY = "0.8083"

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
                      leading=20, textColor=ROSE, spaceBefore=16, spaceAfter=8)
S_H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5,
                      leading=16, textColor=INK, spaceBefore=10, spaceAfter=5)
S_BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5,
                        leading=15.5, textColor=INK, spaceAfter=6)
S_BULLET = ParagraphStyle("bullet", parent=S_BODY, leftIndent=14, bulletIndent=4,
                          spaceAfter=3)
S_CODE = ParagraphStyle("code", fontName="Courier", fontSize=9,
                        leading=12.5, textColor=INK)
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
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def data_table(header, rows):
    data = [[Paragraph(f"<b>{h}</b>", S_BODY) for h in header]]
    for row in rows:
        data.append([Paragraph(c, S_BODY) for c in row])
    t = Table(data, colWidths=[32 * mm, 62 * mm, 76 * mm], repeatRows=1)
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
story.append(Spacer(1, 18 * mm))
story.append(Paragraph("ML Model as an API", S_TITLE))
story.append(Paragraph("Sentiment Classifier", S_TITLE))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph(
    "Train a sentiment classifier and serve it as a live REST API — "
    "built with scikit-learn and FastAPI. No Docker, no PyTorch.", S_SUB))
story.append(Spacer(1, 6 * mm))
story.append(Paragraph("<b>Author:</b> Muhammad Sharjeel Sarwar", S_BODY))
story.append(Paragraph("<b>Date:</b> 1 November 2026", S_BODY))
story.append(Paragraph("<b>Test accuracy:</b> 0.8083 &nbsp;|&nbsp; <b>Model size:</b> 0.23 MB", S_BODY))
story.append(Spacer(1, 6 * mm))

# ---- Architecture flow ----
story.append(Paragraph("Architecture — at a glance", S_H1))
story.append(code_block([
    "Dataset &rarr; Train &rarr; Saved model &rarr; REST API &rarr; Live requests",
    "",
    "UCI Sentiment Labelled Sentences (3,000 reviews)",
    "  &rarr; TF-IDF + Logistic Regression  (train.py)",
    "  &rarr; model/sentiment_model.joblib  (0.23 MB)",
    "  &rarr; FastAPI app  (app.py)  —  POST /predict",
    "  &rarr; anyone can send a request and get a label + score back",
]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph("Example request / response", S_H2))
story.append(code_block([
    "POST /predict",
    '{ "text": "I absolutely loved this movie, it was fantastic!" }',
    "",
    "Response:  { \"label\": \"positive\", \"score\": 0.896 }",
]))
story.append(Spacer(1, 2 * mm))

# ---- What was used ----
story.append(Paragraph("What was used — and why", S_H1))
story.append(data_table(
    ["Tool", "Used for", "Why this one"],
    [
        ["Python 3.12", "Everything", "The language the whole ML + API stack is written in."],
        ["scikit-learn 1.9.1", "TF-IDF + LogisticRegression", "Battle-tested, tiny, fast. Trains in seconds; no GPU needed."],
        ["FastAPI 0.162.0", "The REST API", "Modern, fast, automatic docs at /docs, pydantic validation built in."],
        ["Uvicorn 0.54.0", "Serving the API", "The standard ASGI server for FastAPI; one command to run locally."],
        ["joblib 1.6.0", "Saving / loading the model", "Saves the vectorizer + model as one small file (0.23 MB)."],
        ["pydantic 2.14.2", "Request validation", "Rejects bad input (empty text) with clean 400 errors."],
        ["Mangum 0.22.0", "Vercel serverless adapter", "Wraps the FastAPI app so Vercel's Python runtime can serve it."],
    ]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph("Deliberately NOT used", S_H2))
story.append(Paragraph(
    "<b>Docker — not used.</b> Docker Desktop on Windows needs installation and uses too "
    "much RAM on the laptop. Nothing in this project needs a container: <font face='Courier' size='10'>uvicorn</font> "
    "runs it locally, Render runs native Python, and Vercel runs serverless Python.", S_BULLET, bulletText="•"))
story.append(Paragraph(
    "<b>PyTorch — not used.</b> Far too heavy for a small text classifier. TF-IDF + Logistic "
    "Regression reaches 0.81 accuracy on this dataset, trains in seconds, and the saved model "
    "is 0.23 MB instead of hundreds of MB.", S_BULLET, bulletText="•"))

# ---- Step by step ----
story.append(Paragraph("Step-by-step build", S_H1))

story.append(Paragraph("Step 1 — Dataset", S_H2))
story.append(Paragraph(
    "<b>UCI Sentiment Labelled Sentences</b> (archive.ics.uci.edu, dataset 331): 3,000 labelled "
    "sentences drawn from three sources — Amazon product reviews, IMDb movie reviews and Yelp "
    "restaurant reviews — 1,500 positive and 1,500 negative, perfectly balanced. "
    "<font face='Courier' size='10'>train.py</font> downloads the zip once (~85 KB) and parses the three "
    "labelled files. Split: 80% train (2,400) / 20% test (600), stratified, random_state 42.", S_BODY))

story.append(Paragraph("Step 2 — Text preprocessing", S_H2))
story.append(Paragraph(
    "A <font face='Courier' size='10'>TfidfVectorizer</font> turns each sentence into numbers: lowercase, "
    "English stop-words removed, word unigrams + bigrams, capped at <b>5,000 features</b>. "
    "The cap keeps the model small and fast — the whole saved model is 0.23 MB.", S_BODY))

story.append(Paragraph("Step 3 — Training", S_H2))
story.append(Paragraph(
    "<b>LogisticRegression</b> (max_iter=1000, C=1.0) on the TF-IDF features. Training takes a few "
    "seconds on a laptop. Measured on the held-out 600-sentence test split:", S_BODY))
story.append(data_table(
    ["Metric", "Negative", "Positive"],
    [
        ["Precision", "0.82", "0.80"],
        ["Recall", "0.79", "0.82"],
        ["F1-score", "0.81", "0.81"],
        ["<b>Overall accuracy</b>", "", "<b>0.8083</b>"],
    ]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "Test accuracy 0.8083 — comfortably above the 0.80 target for this dataset and model size.", S_BODY))

story.append(Paragraph("Step 4 — Saving the model", S_H2))
story.append(Paragraph(
    "The fitted vectorizer, the trained model and a small metadata dict (dataset name, accuracy, "
    "hyperparameters) are saved together with <font face='Courier' size='10'>joblib</font> as a single file: "
    "<font face='Courier' size='10'>model/sentiment_model.joblib</font> (0.23 MB). The API loads this one file at "
    "startup — no retraining at serve time.", S_BODY))

story.append(Paragraph("Step 5 — The FastAPI app (app.py)", S_H2))
story.append(Paragraph(
    "The model is loaded <b>once</b> at startup (FastAPI lifespan handler) and reused for every request. "
    "Pydantic models validate all input; empty text returns a 400, never a crash.", S_BODY))
story.append(data_table(
    ["Endpoint", "What it does", "Example"],
    [
        ["GET /", "Project info, model, endpoints", "Returns name, dataset, test accuracy"],
        ["GET /health", "Health check", '{ "status": "ok", "model_loaded": true }'],
        ["POST /predict", "Classify one sentence", '{ "label": "positive", "score": 0.896 }'],
        ["POST /predict/batch", "Classify up to 100 texts", "Returns a list of predictions"],
        ["GET /demo", "In-browser demo page", "Type a sentence, get a result card"],
        ["GET /face", "Face emotion detection page", "Camera or photo, 7 expressions, in-browser"],
        ["GET /docs", "Interactive API docs", "FastAPI's built-in Swagger UI"],
    ]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "The <b>score</b> is the model's probability for the predicted label (0–1, rounded to 4 decimals). "
    "So a positive prediction with score 0.896 means the model is 89.6% confident it is positive.", S_BODY))

story.append(Paragraph("Step 6 — The /demo page", S_H2))
story.append(Paragraph(
    "A polished, self-contained demo page with this project's own fresh identity — soft blush "
    "background (#FDF2F5), deep rose / magenta accents (#D6336C / #E91E63), dark ink text, white "
    "cards with soft shadows and rounded corners. It has a large textarea, an “Analyze sentiment” "
    "button, three clickable example sentences (clearly positive, clearly negative, mixed), a result "
    "card showing the label as a coloured pill plus a confidence meter, a loading state "
    "(“Analyzing…”) while the request runs, and a graceful inline error message if the API fails. "
    "Verified visually at desktop 1280px and mobile 390px — no overflow, no overlapping text, "
    "0 console errors.", S_BODY))

story.append(Paragraph("Step 7 — Local testing (verified)", S_H2))
story.append(Paragraph(
    "Server started with <font face='Courier' size='10'>uvicorn app:app --port 8000</font>; "
    "<font face='Courier' size='10'>test_api.py</font> runs 8 checks — <b>8/8 passed, 0 errors</b>. "
    "The exact curl checks and their real outputs:", S_BODY))
story.append(code_block([
    "$ curl http://127.0.0.1:8000/health",
    '{"status":"ok","model_loaded":true}',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json'",
    "  -d '{\"text\": \"I absolutely loved this movie, it was fantastic and wonderful!\"}'",
    '{"label":"positive","score":0.896}',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json'",
    "  -d '{\"text\": \"This was terrible, a complete waste of time, I hated every minute of it.\"}'",
    '{"label":"negative","score":0.8478}',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict/batch -H 'Content-Type: application/json'",
    "  -d '{\"texts\": [\"Great product, highly recommended!\", \"Awful experience, never again.\"]}'",
    '[{"label":"positive","score":0.8399},{"label":"negative","score":0.7788}]',
    "",
    "$ curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{\"text\": \"   \"}'",
    '{"detail":"text must not be empty"}   (HTTP 400)',
    "",
    "$ curl -o /dev/null -w \"%{http_code}\" http://127.0.0.1:8000/demo   ->  200",
]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "The Mangum / Vercel wrapper was also verified locally: importing <font face='Courier' size='10'>api/index.py</font> "
    "succeeds and a simulated serverless call returns GET /health → 200 "
    "<font face='Courier' size='10'>{\"status\":\"ok\",\"model_loaded\":true}</font>, POST /predict → 200 positive, "
    "empty text → 400.", S_BODY))

story.append(Paragraph("Step 8 — Deployment (both without Docker)", S_H2))
story.append(Paragraph("<b>Render — native Python web service (recommended home):</b>", S_BODY))
story.append(Paragraph("1. Push the folder to a GitHub repo.", S_BULLET, bulletText="1."))
story.append(Paragraph("2. Render dashboard → New → Web Service → pick the repo.", S_BULLET, bulletText="2."))
story.append(Paragraph("3. Environment: Python.", S_BULLET, bulletText="3."))
story.append(Paragraph("4. Build Command: <font face='Courier' size='10'>pip install -r requirements.txt &amp;&amp; python train.py</font>", S_BULLET, bulletText="4."))
story.append(Paragraph("5. Start Command: <font face='Courier' size='10'>uvicorn app:app --host 0.0.0.0 --port $PORT</font>", S_BULLET, bulletText="5."))
story.append(Paragraph(
    "The model is trained during the build, so the model file is not committed to the repo — "
    "it is regenerated by train.py on every deploy.", S_BODY))
story.append(Paragraph(
    "Free tier works fine. Note: a free Render service sleeps when idle, so the first request after "
    "a sleep takes a few seconds while it wakes up and reloads the model.", S_BODY))
story.append(Paragraph("<b>Vercel — serverless (same repo):</b>", S_BODY))
story.append(Paragraph("1. Vercel dashboard → Add New → Project → import the same repo (or <font face='Courier' size='10'>vercel</font> from the CLI).", S_BULLET, bulletText="1."))
story.append(Paragraph("2. Vercel auto-detects Python and installs <font face='Courier' size='10'>requirements.txt</font>; <font face='Courier' size='10'>vercel.json</font> sets the build command to install requirements and run <font face='Courier' size='10'>train.py</font>, so the model is generated during the build.", S_BULLET, bulletText="2."))
story.append(Paragraph("3. <font face='Courier' size='10'>api/index.py</font> wraps the FastAPI app with Mangum; <font face='Courier' size='10'>vercel.json</font> rewrites every path to <font face='Courier' size='10'>/api/index.py</font>.", S_BULLET, bulletText="3."))
story.append(Paragraph(
    "<b>Render vs Vercel:</b> Render is a persistent server — the model loads once and stays loaded, "
    "so every request is fast. Vercel is serverless — cold starts reload the model (small at 0.23 MB, "
    "about a second) — but it keeps everything in one place he already uses.", S_BODY))

story.append(Paragraph("Step 9 — Face emotion detection (GET /face)", S_H1))
story.append(Paragraph(
    "<b>What was added.</b> A second demo page, <font face='Courier' size='10'>GET /face</font>, in the same "
    "app and the same pink/rose identity as <font face='Courier' size='10'>/demo</font>. It reads facial "
    "expressions two ways: a <b>live camera</b> (Start/Stop buttons, front camera via "
    "<font face='Courier' size='10'>getUserMedia</font> with <font face='Courier' size='10'>facingMode: \"user\"</font>, "
    "face box + top-expression label drawn live on a canvas overlay) and a <b>photo upload</b> (pick a photo "
    "from the gallery, same detection on the still image). The result shows the top emotion big with its "
    "emoji and a confidence bar, plus all 7 expression scores listed underneath. The two demo pages link "
    "to each other (“Try face emotion &rarr;” / “&larr; Back to text demo”).", S_BODY))
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
    "The seven expressions and how they are shown:", S_BODY))
story.append(data_table(
    ["Model output", "Shown on the page as", ""],
    [
        ["happy", "Happy", ""],
        ["sad", "Sad / Hurting", ""],
        ["angry", "Angry", ""],
        ["fearful", "Scared", ""],
        ["disgusted", "Disgusted", ""],
        ["surprised", "Surprised", ""],
        ["neutral", "Neutral", ""],
    ]))
story.append(Spacer(1, 2 * mm))
story.append(Paragraph(
    "<b>Why it runs in the browser.</b> The Render free server cannot run a face model — it is a small "
    "Python service with no GPU and no camera. Doing detection client-side also means <b>camera frames "
    "and photos never leave the user's device</b>: nothing is uploaded, nothing is sent to the server, "
    "and no image is ever stored. The TensorFlow.js backend is WebGL when the device supports it, "
    "with an automatic fallback to CPU.", S_BODY))
story.append(Paragraph(
    "<b>Graceful failures.</b> Camera permission denied or no camera on the device shows a clear inline "
    "message and points to photo upload instead; a photo with no face says so plainly; if the library or "
    "the models fail to load from the CDN, the page shows an inline error instead of a blank page. The "
    "page renders immediately even on a slow connection (the CDN script is deferred).", S_BODY))
story.append(Paragraph(
    "<b>Honest limits.</b> The model detects the 7 standard expressions only. “Hurting” is <b>not</b> a "
    "separate class — it is shown as <b>Sad / Hurting</b>, i.e. the model's <i>sad</i> score read in plain "
    "words, exactly like the feeling line on the text demo. A smile scores as happy; a frown or tears "
    "score as sad. The expressions are a reading of the face, not a diagnosis of how someone feels.", S_BODY))
story.append(Paragraph(
    "<b>What was verified (and what was not).</b> <font face='Courier' size='10'>GET /face</font> returns 200 locally; "
    "the library script, all three model manifests and all three model weight files return HTTP 200 "
    "(checked with curl). Detection was run end-to-end in a headless browser on a real public-domain "
    "smiling portrait (Wikimedia Commons): top emotion <b>Happy, score 1.00</b>, all 7 scores listed, face "
    "box + landmarks drawn on the overlay (verified geometrically: the canvas exactly matches the displayed "
    "photo size, box inside the bounds). Layout is clean at desktop 1280px and mobile 390px (0 px horizontal "
    "overflow), 0 console errors. The camera-denied path was exercised and shows its message correctly. "
    "<b>Not verified:</b> the live camera on a real phone — headless Firefox has no camera, so that path "
    "still needs a real-phone check. Also note: the sandbox's headless Firefox could not fetch jsDelivr "
    "directly (its requests hang), so the in-browser detection test used byte-identical local copies of the "
    "same pinned files; the shipped page still loads everything from the CDN, where every URL was verified "
    "HTTP 200. On the headless test machine TensorFlow.js fell back to the CPU backend (no WebGL there); "
    "on a real phone it will use WebGL.", S_BODY))

# ---- How to run + file list ----
story.append(Paragraph("How to run it locally", S_H1))
story.append(code_block([
    "pip install -r requirements.txt",
    "python train.py          # trains the model, saves model/sentiment_model.joblib",
    "uvicorn app:app --reload # API at http://127.0.0.1:8000",
    "python test_api.py       # 8 smoke tests against the running server",
]))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph("File list", S_H1))
story.append(data_table(
    ["File", "What it is", "Size / notes"],
    [
        ["train.py", "Trains TF-IDF + LogisticRegression", "Downloads UCI data, prints accuracy"],
        ["app.py", "FastAPI app + /demo and /face pages", "7 endpoints incl. /face, loads model once"],
        ["api/index.py", "Vercel entrypoint (Mangum wrapper)", "No change to app.py needed"],
        ["vercel.json", "Routes all paths to /api/index.py", "Vercel only"],
        ["requirements.txt", "fastapi, uvicorn, scikit-learn, joblib, pydantic, mangum", "Pinned versions"],
        ["test_api.py", "8 smoke tests (health, predict, batch, demo)", "8/8 passed"],
        ["README.md", "Setup, curl examples, Render + Vercel deploys", "—"],
        ["shots/face-desktop.png", "GET /face verified at 1280px", "Happy 1.00, box on face"],
        ["shots/face-mobile.png", "GET /face verified at 390px", "0 px overflow"],
        ["model/sentiment_model.joblib", "Trained vectorizer + model", "0.23 MB"],
        ["data/sentiment_labelled_sentences.zip", "UCI dataset (downloaded by train.py)", "~85 KB"],
        ["ML_Model_API_Build_Report.pdf", "This report", "—"],
    ]))

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=20 * mm,
                      title="ML Model as an API — Sentiment Classifier — Build Report",
                      author="Muhammad Sharjeel Sarwar")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print(f"Built {OUT}")
