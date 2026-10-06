from fastapi import FastAPI
from vaani.app.api import chat, voice, tools
from vaani.app.rag.ingest import ingest_knowledge_base
import os

app = FastAPI(title="Vaani API", description="Multilingual Agentic Voice AI Backend")

# Ingest KB on startup if Chroma DB is empty or as a setup step
@app.on_event("startup")
async def startup_event():
    if not os.path.exists("vaani/chroma_db"):
        print("Initializing Knowledge Base...")
        ingest_knowledge_base()

app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(voice.router, prefix="/api", tags=["Voice"])
app.include_router(tools.router, prefix="/api", tags=["Tools"])

@app.get("/api/health")
async def health():
    return {"status": "healthy"}
