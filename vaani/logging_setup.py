"""Process-wide logging configuration."""

from __future__ import annotations

import logging

from vaani.config import get_settings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=get_settings().log_level, format=_FORMAT, force=True)
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
