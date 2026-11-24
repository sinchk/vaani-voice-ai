"""Request / response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vaani.agent.agent import AgentResult


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    content: str
    source: str
    confidence: float


class ToolCallInfo(BaseModel):
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    response: str
    sources: list[Source] = []
    tool_calls: list[ToolCallInfo] = []
    confidence: float | None = Field(
        None, description="Similarity of the best knowledge-base chunk used; null if none was used."
    )
    handoff: bool = False
    handoff_summary: dict[str, Any] = {}

    @classmethod
    def from_result(cls, r: AgentResult) -> ChatResponse:
        return cls(
            response=r.response,
            sources=[Source(**s.to_dict()) for s in r.sources],
            tool_calls=[ToolCallInfo(**c) for c in r.tool_calls],
            confidence=r.confidence,
            handoff=r.handoff,
            handoff_summary=r.handoff_summary,
        )


class VoiceResponse(ChatResponse):
    transcript: str
    detected_language: str
    audio_url: str | None = Field(None, description="Relative URL of the spoken reply (mp3), if synthesis worked.")
