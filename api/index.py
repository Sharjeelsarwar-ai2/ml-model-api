"""
api/index.py — Vercel serverless entrypoint.

Vercel's Python runtime serves each file under `api/` as a serverless function.
This module wraps the FastAPI app from app.py with Mangum (an ASGI adapter)
so the same API that runs under `uvicorn` locally also runs on Vercel.

No changes to app.py are needed: the model loads once at startup via the
FastAPI lifespan handler, which Mangum triggers on the first (cold) request.
"""

import os
import sys

# Make the project root importable so `from app import app` works
# whether Vercel runs this from api/ or from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangum import Mangum  # noqa: E402

from app import app  # noqa: E402

# Vercel calls this handler for every request routed to /api/index.py.
handler = Mangum(app, lifespan="auto")
