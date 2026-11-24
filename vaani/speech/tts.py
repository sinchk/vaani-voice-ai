"""Text-to-speech with gTTS (English and Kannada)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path

from gtts import gTTS

from vaani.config import get_settings
from vaani.text_utils import detect_language

logger = logging.getLogger(__name__)

AUDIO_NAME_RE = re.compile(r"^response_[0-9a-f-]{36}\.mp3$")


class TTSProcessor:
    def __init__(self, output_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.audio_dir)
        self.ttl_seconds = settings.audio_ttl_seconds if ttl_seconds is None else ttl_seconds
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, lang: str | None = None) -> str | None:
        """Render ``text`` to an mp3 and return its file *name*, or None on failure.

        ``lang`` defaults to the script of the text (Kannada vs English).
        """
        lang = lang or detect_language(text)
        self._purge_expired()
        name = f"response_{uuid.uuid4()}.mp3"
        try:
            gTTS(text=text, lang=lang).save(str(self.output_dir / name))
        except Exception:  # noqa: BLE001 - gTTS needs network; voice reply degrades to text-only
            logger.exception("TTS failed")
            return None
        return name

    def path_for(self, name: str) -> Path | None:
        """Resolve a previously returned file name; None if malformed or missing."""
        if not AUDIO_NAME_RE.fullmatch(name):
            return None
        path = self.output_dir / name
        return path if path.is_file() else None

    def _purge_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for f in self.output_dir.glob("response_*.mp3"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
