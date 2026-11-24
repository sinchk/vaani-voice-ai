"""Speech-to-text with faster-whisper (multilingual, handles English and Kannada)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from vaani.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    probability: float


class ASRProcessor:
    """Loads the Whisper model lazily so importing the API (and running tests) stays fast."""

    def __init__(self, model_size: str | None = None) -> None:
        self.model_size = model_size or get_settings().whisper_model
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                s = get_settings()
                logger.info("Loading Whisper '%s' on %s", self.model_size, s.whisper_device)
                self._model = WhisperModel(
                    self.model_size, device=s.whisper_device, compute_type=s.whisper_compute_type
                )
        return self._model

    def transcribe(self, audio_path: str) -> Transcript:
        segments, info = self._load().transcribe(audio_path, beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(text=text, language=info.language, probability=float(info.language_probability))
