"""In-process session memory: bounded conversation history plus extracted slots."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    messages: list[dict[str, str]] = field(default_factory=list)  # {"role": "user"|"model", "text": ...}
    slots: dict[str, Any] = field(default_factory=dict)  # e.g. last_order_id, customer


class SessionMemory:
    """Thread-safe store. Old turns and idle sessions are evicted so memory stays bounded."""

    def __init__(self, max_messages: int = 20, max_sessions: int = 1000) -> None:
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _get(self, session_id: str) -> Session:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        else:
            self._sessions[session_id] = Session()
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return self._sessions[session_id]

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._get(session_id).messages)

    def add_message(self, session_id: str, role: str, text: str) -> None:
        with self._lock:
            session = self._get(session_id)
            session.messages.append({"role": role, "text": text})
            # Keep the most recent window, always starting on a user turn.
            if len(session.messages) > self.max_messages:
                session.messages = session.messages[-self.max_messages :]
                while session.messages and session.messages[0]["role"] != "user":
                    session.messages.pop(0)

    def update_slot(self, session_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._get(session_id).slots[key] = value

    def get_slots(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._get(session_id).slots)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
