"""Shared fixtures. Nothing here touches the network or the real data directory."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from google.genai import types

from vaani.agent.agent import VaaniAgent
from vaani.agent.llm import LLMTurn, ToolCall
from vaani.agent.memory import SessionMemory
from vaani.rag.retriever import RetrievedChunk
from vaani.tools import ToolRegistry
from vaani.tools.appointments import AppointmentService
from vaani.tools.orders import OrderService
from vaani.tools.store import JsonStore

ORDERS = [
    {"order_id": "1042", "status": "Shipped", "customer": "Sinchana", "items": ["Headphones"], "eta": "2030-01-01"},
    {"order_id": "1043", "status": "Processing", "customer": "Rahul", "items": ["Case"], "eta": "2030-01-02"},
    {"order_id": "1044", "status": "Cancelled", "customer": "Ananya", "items": ["Shoes"], "eta": "2030-01-03"},
]
FUTURE_DAY = (date.today() + timedelta(days=30)).isoformat()
APPOINTMENTS = [{"date": FUTURE_DAY, "time": "11:00 AM", "customer": "Sinchana", "status": "Booked"}]


def _store(tmp_path, name, records) -> JsonStore:
    seed = tmp_path / "seed" / name
    seed.parent.mkdir(exist_ok=True)
    seed.write_text(json.dumps(records), encoding="utf-8")
    return JsonStore(seed, tmp_path / "state" / name)


@pytest.fixture
def order_service(tmp_path) -> OrderService:
    return OrderService(_store(tmp_path, "orders.json", ORDERS))


@pytest.fixture
def appointment_service(tmp_path) -> AppointmentService:
    return AppointmentService(_store(tmp_path, "appointments.json", APPOINTMENTS))


@pytest.fixture
def registry(order_service, appointment_service) -> ToolRegistry:
    return ToolRegistry(order_service, appointment_service)


class FakeLLM:
    """Replays scripted turns and records what the agent sent."""

    def __init__(self, *turns: LLMTurn) -> None:
        self.turns = list(turns)
        self.calls: list[dict] = []

    def generate(self, system, contents, tools):
        self.calls.append({"system": system, "contents": list(contents), "tools": tools})
        return self.turns.pop(0)


class FakeRetriever:
    def __init__(self, chunks=None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error

    def retrieve(self, query):
        if self.error:
            raise self.error
        return self.chunks


def text_turn(text: str) -> LLMTurn:
    return LLMTurn(text=text, content=types.Content(role="model", parts=[types.Part(text=text)]))


def call_turn(name: str, **args) -> LLMTurn:
    return LLMTurn(
        text="",
        tool_calls=[ToolCall(name, args)],
        content=types.Content(role="model", parts=[types.Part.from_function_call(name=name, args=args)]),
    )


def make_agent(registry, llm, retriever=None) -> VaaniAgent:
    return VaaniAgent(llm=llm, retriever=retriever or FakeRetriever(), tools=registry, memory=SessionMemory())


@pytest.fixture
def return_chunk() -> RetrievedChunk:
    return RetrievedChunk("Customers can return products within 30 days.", "return_policy.txt", 0.82)
