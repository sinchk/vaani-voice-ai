from fastapi import APIRouter, UploadFile, File, Form
from vaani.app.speech.asr import ASRProcessor
from vaani.app.speech.tts import TTSProcessor
from vaani.app.agents.agent import VaaniAgent
import shutil
import os

router = APIRouter()
asr = ASRProcessor()
tts = TTSProcessor()
agent = VaaniAgent()

@router.post("/voice")
async def voice_endpoint(session_id: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. ASR
        asr_result = asr.transcribe(temp_path)
        user_text = asr_result["text"]
        
        # 2. Agent
        agent_result = agent.chat(session_id, user_text)
        
        # 3. TTS
        # Determine language for TTS - simplified: use ASR detected lang or default to 'en'
        lang = 'kn' if asr_result["language"] == 'kn' else 'en'
        audio_path = tts.synthesize(agent_result["response"], lang=lang)
        
        return {
            "transcript": user_text,
            "response": agent_result["response"],
            "audio_url": audio_path,
            "sources": agent_result["sources"],
            "handoff": agent_result["handoff"]
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
