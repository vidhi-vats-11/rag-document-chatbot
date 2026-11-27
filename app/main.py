"""Application entry point: creates the FastAPI app, wires services, registers routes.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import chat, documents, health, models
from app.core.config import Settings, get_settings
from app.core.container import Container, build_container
from app.core.exceptions import AppError
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """App factory. Tests pass their own settings/container; production uses defaults."""
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "container", None) is None:
            app.state.container = build_container(settings)
            logger.info("Loading embedding model (first run downloads it)...")
            app.state.container.embedder.warm_up()
            if settings.rerank_enabled:
                logger.info("Loading re-ranking model (first run downloads it)...")
                app.state.container.reranker.warm_up()
        c: Container = app.state.container
        resumed = c.document_service.resume_unfinished()
        if resumed:
            logger.info("Resumed indexing of %d unfinished document(s).", resumed)
        if not c.extractor.ocr_available():
            logger.warning("Tesseract OCR not found: scanned PDFs and images cannot be read.")
        logger.info(
            "Ready. LLM provider=%s model=%s configured=%s",
            settings.llm_provider.value, settings.llm_model, c.chat_service.generation.configured,
        )
        yield
        c.document_service.shutdown()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Upload PDFs, scanned PDFs, images and DOCX files, then ask grounded questions about them.",
        lifespan=lifespan,
    )
    app.state.container = container

    _register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(chat.router, prefix=API_PREFIX)
    app.include_router(models.router, prefix=API_PREFIX)

    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=APP_DIR / "templates")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "app_name": settings.app_name,
                "max_upload_mb": settings.max_upload_size_mb,
                "asset_version": _asset_version(),
            },
        )

    return app


def _asset_version() -> str:
    """Changes whenever app.js or style.css changes.

    It is added to their URLs (app.js?v=...), so the browser downloads the new file
    instead of running an old copy it saved earlier (its "cache").
    """
    files = [APP_DIR / "static" / "js" / "app.js", APP_DIR / "static" / "css" / "style.css"]
    return str(max(int(f.stat().st_mtime) for f in files))


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        # Our own errors carry safe, user-facing messages.
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.error_code, "message": exc.message}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'] if p != 'body')}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse(
            status_code=422, content={"error": "validation_error", "message": details or "Invalid request."}
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback for developers; never leak it to users.
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500, content={"error": "internal_error", "message": "An unexpected error occurred."}
        )


app = create_app()
