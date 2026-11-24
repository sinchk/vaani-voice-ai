"""The Vaani agent: RAG context + LLM function calling + session memory + human handoff."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from vaani.agent.handoff import wants_human
from vaani.agent.llm import LLM, GeminiLLM, LLMUnavailableError
from vaani.agent.memory import SessionMemory
from vaani.agent.prompts import HANDOFF_REPLY, build_system_prompt, build_user_message
from vaani.config import get_settings
from vaani.rag.retriever import KBRetriever, RetrievedChunk
from vaani.text_utils import detect_language
from vaani.tools import HANDOFF_TOOL, ToolRegistry

logger = logging.getLogger(__name__)

# Tool argument -> conversation slot, so the handoff summary knows what was discussed.
_SLOT_FOR_ARG = {"order_id": "last_order_id", "customer": "customer", "date": "last_date", "time": "last_time"}

_FALLBACK_REPLY = {
    "en": "Sorry, I couldn't complete that. Could you rephrase, or would you like a human agent?",
    "kn": "ಕ್ಷಮಿಸಿ, ಇದನ್ನು ಪೂರ್ಣಗೊಳಿಸಲು ಆಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳುವಿರಾ, ಅಥವಾ ಮಾನವ ಏಜೆಂಟ್ ಬೇಕೇ?",
}


@dataclass
class AgentResult:
    response: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {"name", "args", "result"}
    # Similarity of the best knowledge-base chunk; None when the answer used no retrieved context.
    confidence: float | None = None
    handoff: bool = False
    handoff_summary: dict[str, Any] = field(default_factory=dict)


class VaaniAgent:
    def __init__(
        self,
        llm: LLM | None = None,
        retriever: KBRetriever | None = None,
        tools: ToolRegistry | None = None,
        memory: SessionMemory | None = None,
    ) -> None:
        settings = get_settings()
        self.llm = llm or GeminiLLM()
        self.retriever = retriever or KBRetriever()
        self.tools = tools or ToolRegistry.default()
        self.memory = memory or SessionMemory(max_messages=settings.max_history_messages)
        self.max_tool_rounds = settings.max_tool_rounds

    # ------------------------------------------------------------------ public

    def chat(self, session_id: str, user_input: str) -> AgentResult:
        sources = self._retrieve(user_input)
        contents = self._history_contents(session_id)
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=build_user_message(user_input, [s.content for s in sources]))],
            )
        )

        tool_calls: list[dict[str, Any]] = []
        handoff_reason: str | None = None
        try:
            reply, handoff_reason = self._run_tool_loop(session_id, contents, tool_calls)
        except LLMUnavailableError:
            logger.exception("LLM unavailable for session %s", session_id)
            if not wants_human(user_input):
                raise
            reply = ""  # fall through to the canned handoff reply below

        if handoff_reason is None and wants_human(user_input):
            handoff_reason = "Customer asked for a human agent."
        handoff = handoff_reason is not None
        if handoff and not reply:
            reply = HANDOFF_REPLY[detect_language(user_input)]
        if not reply:
            reply = _FALLBACK_REPLY[detect_language(user_input)]

        self.memory.add_message(session_id, "user", user_input)
        self.memory.add_message(session_id, "model", reply)

        return AgentResult(
            response=reply,
            sources=sources,
            tool_calls=tool_calls,
            confidence=sources[0].confidence if sources else None,
            handoff=handoff,
            handoff_summary=self._handoff_summary(session_id, handoff_reason, tool_calls) if handoff else {},
        )

    # ----------------------------------------------------------------- helpers

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        try:
            return self.retriever.retrieve(query)
        except Exception:  # noqa: BLE001 - degrade gracefully: tools and chat still work without RAG
            logger.exception("Retrieval failed; continuing without knowledge-base context")
            return []

    def _history_contents(self, session_id: str) -> list[types.Content]:
        return [
            types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
            for m in self.memory.get_messages(session_id)
        ]

    def _run_tool_loop(
        self, session_id: str, contents: list[types.Content], tool_calls: list[dict[str, Any]]
    ) -> tuple[str, str | None]:
        """Call the LLM until it answers in text. Returns (reply, handoff_reason)."""
        system = build_system_prompt()
        handoff_reason: str | None = None
        for _ in range(self.max_tool_rounds):
            turn = self.llm.generate(system, contents, self.tools.declarations())
            if not turn.tool_calls:
                return turn.text, handoff_reason

            if turn.content is not None:
                contents.append(turn.content)  # echo the model's call (keeps thought signatures intact)
            response_parts = []
            for call in turn.tool_calls:
                result = self.tools.execute(call.name, call.args)
                logger.info("tool %s(%s) -> %s", call.name, call.args, result)
                tool_calls.append({"name": call.name, "args": call.args, "result": result})
                self._remember_slots(session_id, call.args)
                if call.name == HANDOFF_TOOL:
                    handoff_reason = str(call.args.get("reason") or "Escalated by assistant.")
                response_parts.append(types.Part.from_function_response(name=call.name, response=result))
            contents.append(types.Content(role="user", parts=response_parts))

        logger.warning("Tool loop hit the %d-round limit", self.max_tool_rounds)
        return "", handoff_reason

    def _remember_slots(self, session_id: str, args: dict[str, Any]) -> None:
        for arg, slot in _SLOT_FOR_ARG.items():
            if args.get(arg):
                self.memory.update_slot(session_id, slot, args[arg])

    def _handoff_summary(self, session_id: str, reason: str | None, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        recent = self.memory.get_messages(session_id)[-6:]
        return {
            "reason": reason,
            "slots": self.memory.get_slots(session_id),
            "tools_used": [c["name"] for c in tool_calls if c["name"] != HANDOFF_TOOL],
            "recent_messages": recent,
        }
