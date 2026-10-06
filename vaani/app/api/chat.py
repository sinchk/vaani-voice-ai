from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from vaani.app.agents.agent import VaaniAgent

router = APIRouter()
agent = VaaniAgent()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        result = agent.chat(req.session_id, req.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
