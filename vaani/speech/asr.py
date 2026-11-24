"""Speech-to-text with faster-whisper (multilingual, handles English and Kannada)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from vaani.config import get_settings
from vaani.text_utils import detect_language

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


_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}
_ASR_PROMPT = (
    "Transcribe this audio exactly as spoken, in the original language and script "
    "(Kannada in Kannada script, English in English, mixed speech kept mixed). "
    "Write digits as numerals. Output only the transcript, or nothing if there is no speech."
)


class GeminiASR:
    """Transcription via Gemini's native audio understanding (much stronger on Kannada than small Whisper)."""

    def __init__(self, models: list[str] | None = None) -> None:
        s = get_settings()
        self.models = models or [s.chat_model, *s.fallback_models]

    def transcribe(self, audio_path: str) -> Transcript:
        from google.genai import types

        from vaani.agent.llm import LLMUnavailableError, generate_with_retry

        path = Path(audio_path)
        part = types.Part.from_bytes(data=path.read_bytes(), mime_type=_MIME.get(path.suffix.lower(), "audio/wav"))
        config = types.GenerateContentConfig(temperature=0.0)
        for model in self.models:
            try:
                response = generate_with_retry(model, [part, _ASR_PROMPT], config, retries=1)
                break
            except LLMUnavailableError:
                logger.warning("ASR: %s unavailable; trying next model", model)
        else:
            raise LLMUnavailableError("All Gemini models are unavailable for transcription")
        text = (response.text or "").strip()
        return Transcript(text=text, language=detect_language(text), probability=1.0)


def build_asr() -> ASRProcessor | GeminiASR:
    return GeminiASR() if get_settings().asr_backend == "gemini" else ASRProcessor()
