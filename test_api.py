"""
test_api.py — smoke tests for the running API (expects the server on localhost:8000).

Usage:
    uvicorn app:app --port 8000 &
    python test_api.py
"""

import urllib.request
import urllib.error
import json

BASE = "http://127.0.0.1:8000"


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    checks = []

    status, body = call("GET", "/health")
    checks.append(("GET /health -> 200 + ok", status == 200 and json.loads(body)["status"] == "ok"))

    status, body = call("POST", "/predict", {"text": "I absolutely loved this movie, it was fantastic and wonderful!"})
    data = json.loads(body)
    checks.append(("positive sentence -> positive, score >= 0.8",
                   status == 200 and data["label"] == "positive" and data["score"] >= 0.8))

    status, body = call("POST", "/predict", {"text": "This was terrible, a complete waste of time, I hated every minute of it."})
    data = json.loads(body)
    checks.append(("negative sentence -> negative, score >= 0.8",
                   status == 200 and data["label"] == "negative" and data["score"] >= 0.8))

    status, body = call("POST", "/predict", {"text": "   "})
    checks.append(("empty text -> 400", status == 400))

    status, body = call("POST", "/predict", {"text": ""})
    checks.append(("missing/empty string -> 400/422", status in (400, 422)))

    status, body = call("POST", "/predict/batch",
                        {"texts": ["Great product, highly recommended!", "Awful experience, never again."]})
    data = json.loads(body)
    checks.append(("batch -> [positive, negative]",
                   status == 200 and [d["label"] for d in data] == ["positive", "negative"]))

    status, body = call("GET", "/demo")
    checks.append(("GET /demo -> 200 HTML with textarea + Analyze button",
                   status == 200 and "<textarea" in body and "Analyze sentiment" in body))

    status, body = call("GET", "/")
    checks.append(("GET / -> 200 with endpoints",
                   status == 200 and "predict" in body))

    failed = 0
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
