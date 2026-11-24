from fastapi import APIRouter, Depends

from vaani.agent.agent import VaaniAgent
from vaani.api.deps import get_agent
from vaani.api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Send a text message to Vaani")
def chat(req: ChatRequest, agent: VaaniAgent = Depends(get_agent)) -> ChatResponse:
    # Sync endpoint: FastAPI runs it in a worker thread, so blocking LLM calls don't stall the event loop.
    return ChatResponse.from_result(agent.chat(req.session_id, req.message))
