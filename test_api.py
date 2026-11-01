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


class NoFollow(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), {k.lower(): v for k, v in e.headers.items()}


def call_nofollow(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(NoFollow())
    try:
        with opener.open(req, timeout=10) as resp:
            return resp.status, resp.read().decode(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), {k.lower(): v for k, v in e.headers.items()}


def main():
    checks = []

    status, body, _ = call("GET", "/health")
    checks.append(("GET /health -> 200 + ok", status == 200 and json.loads(body)["status"] == "ok"))

    status, body, _ = call("POST", "/predict", {"text": "I absolutely loved this movie, it was fantastic and wonderful!"})
    data = json.loads(body)
    checks.append(("positive sentence -> positive, score >= 0.8",
                   status == 200 and data["label"] == "positive" and data["score"] >= 0.8))

    status, body, _ = call("POST", "/predict", {"text": "This was terrible, a complete waste of time, I hated every minute of it."})
    data = json.loads(body)
    checks.append(("negative sentence -> negative, score >= 0.8",
                   status == 200 and data["label"] == "negative" and data["score"] >= 0.8))

    status, body, _ = call("POST", "/predict", {"text": "   "})
    checks.append(("empty text -> 400", status == 400))

    status, body, _ = call("POST", "/predict", {"text": ""})
    checks.append(("missing/empty string -> 400/422", status in (400, 422)))

    status, body, _ = call("POST", "/predict/batch",
                        {"texts": ["Great product, highly recommended!", "Awful experience, never again."]})
    data = json.loads(body)
    checks.append(("batch -> [positive, negative]",
                   status == 200 and [d["label"] for d in data] == ["positive", "negative"]))

    status, body, _ = call("GET", "/demo")
    checks.append(("GET /demo -> 200 HTML with both tabs + text tool",
                   status == 200 and "<textarea" in body and "Analyze sentiment" in body
                   and 'id="tab-text"' in body and 'id="tab-face"' in body
                   and 'id="panel-text"' in body and 'id="panel-face"' in body
                   and "Start camera" in body and "Upload a photo" in body))

    status, body, headers = call_nofollow("GET", "/face")
    checks.append(("GET /face -> 307 to /demo#face",
                   status == 307 and headers.get("location") == "/demo#face"))

    status, body, headers = call_nofollow("GET", "/")
    checks.append(("GET / -> 307 to /demo",
                   status == 307 and headers.get("location") == "/demo"))

    failed = 0
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
