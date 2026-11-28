"""Entry point for Vercel, which looks for an `app` object in a file like index.py.

Locally, keep using:  uvicorn app.main:app --reload
"""

from app.main import app  # noqa: F401
