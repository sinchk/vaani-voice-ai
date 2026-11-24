import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from vaani.agent.agent import VaaniAgent
from vaani.api.deps import get_agent, get_asr, get_tts
from vaani.api.schemas import ChatResponse, VoiceResponse
from vaani.config import get_settings
from vaani.speech.asr import ASRProcessor, GeminiASR
from vaani.speech.tts import TTSProcessor
from vaani.text_utils import detect_language

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}


@router.post("/voice", response_model=VoiceResponse, summary="Speak to Vaani (audio in, audio + text out)")
def voice(
    session_id: str = Form(..., min_length=1, max_length=128),
    file: UploadFile = File(...),
    agent: VaaniAgent = Depends(get_agent),
    asr: ASRProcessor | GeminiASR = Depends(get_asr),
    tts: TTSProcessor = Depends(get_tts),
) -> VoiceResponse:
    suffix = Path(file.filename or "").suffix.lower() or ".wav"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported audio type '{suffix}'. Use one of {sorted(ALLOWED_SUFFIXES)}.")

    limit = get_settings().max_upload_bytes
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Audio exceeds the {limit // (1024 * 1024)} MB limit.")

    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        transcript = asr.transcribe(tmp_name)
    finally:
        os.unlink(tmp_name)

    if not transcript.text:
        raise HTTPException(422, "No speech detected in the audio.")

    result = agent.chat(session_id, transcript.text)
    audio_name = tts.synthesize(result.response, lang=detect_language(result.response))
    return VoiceResponse(
        **ChatResponse.from_result(result).model_dump(),
        transcript=transcript.text,
        detected_language=transcript.language,
        audio_url=f"/api/audio/{audio_name}" if audio_name else None,
    )


@router.get("/audio/{name}", summary="Fetch a synthesized reply", response_class=FileResponse)
def audio(name: str, tts: TTSProcessor = Depends(get_tts)) -> FileResponse:
    path = tts.path_for(name)
    if path is None:
        raise HTTPException(404, "Audio not found (it may have expired).")
    return FileResponse(path, media_type="audio/mpeg")
