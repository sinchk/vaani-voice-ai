"""FastAPI application factory.  Run with:  uvicorn vaani.api.app:app"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vaani import __version__
from vaani.agent.llm import LLMError, LLMUnavailableError
from vaani.api import chat, tools, voice
from vaani.api.deps import get_agent
from vaani.config import get_settings
from vaani.gemini import MissingAPIKeyError
from vaani.logging_setup import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    try:
        get_agent()  # builds the retriever and ingests the knowledge base on first run
        logger.info("Vaani ready")
    except MissingAPIKeyError as exc:
        logger.error("%s - the API will start but chat endpoints will return 503.", exc)
    except Exception:  # noqa: BLE001 - keep the API up so /api/health can report the problem
        logger.exception("Agent failed to initialise")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vaani API",
        description="Multilingual (English / Kannada) agentic voice assistant backend.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(voice.router, prefix="/api", tags=["Voice"])
    app.include_router(tools.router, prefix="/api", tags=["Tools"])

    @app.get("/api/health", tags=["Meta"])
    def health() -> dict:
        s = get_settings()
        return {
            "status": "ok",
            "version": __version__,
            "model": s.chat_model,
            "api_key_configured": bool(s.gemini_api_key),
        }

    def _error(status: int, message: str):
        def handler(_req: Request, exc: Exception) -> JSONResponse:
            logger.error("%s: %s", type(exc).__name__, exc)  # keep the underlying cause in the server log
            return JSONResponse(status_code=status, content={"detail": message})

        return handler

    app.add_exception_handler(MissingAPIKeyError, _error(503, "Server is missing GEMINI_API_KEY."))
    app.add_exception_handler(LLMUnavailableError, _error(503, "The language model is busy. Please retry shortly."))
    app.add_exception_handler(LLMError, _error(502, "The language model request failed."))

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):  # noqa: ARG001
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
