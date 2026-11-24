"""GeminiLLM retry / fallback behaviour, with the SDK call faked out."""

import pytest
from google.genai import errors

from vaani.agent import llm as llm_module
from vaani.agent.llm import GeminiLLM, LLMError, LLMUnavailableError


class FakeResponse:
    candidates = []
    function_calls = None


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


class FakeClient:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        result = self.behaviour(model, len(self.calls))
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)


def install(monkeypatch, behaviour):
    client = FakeClient(behaviour)
    monkeypatch.setattr(llm_module, "get_client", lambda: client)
    return client


def test_retries_transient_errors_then_succeeds(monkeypatch):
    client = install(monkeypatch, lambda m, n: api_error(503) if n < 3 else FakeResponse())
    GeminiLLM("primary", max_retries=3, fallback_model="").generate("sys", [], [])
    assert client.calls == ["primary"] * 3


def test_falls_back_to_secondary_model(monkeypatch):
    client = install(monkeypatch, lambda m, n: api_error(503) if m == "primary" else FakeResponse())
    GeminiLLM("primary", max_retries=1, fallback_model="backup").generate("sys", [], [])
    assert client.calls == ["primary", "primary", "backup"]


def test_raises_unavailable_when_everything_is_down(monkeypatch):
    install(monkeypatch, lambda m, n: api_error(503))
    with pytest.raises(LLMUnavailableError):
        GeminiLLM("primary", max_retries=1, fallback_model="backup").generate("sys", [], [])


def test_client_errors_are_not_retried(monkeypatch):
    client = install(monkeypatch, lambda m, n: api_error(400))
    with pytest.raises(LLMError) as exc:
        GeminiLLM("primary", max_retries=3, fallback_model="backup").generate("sys", [], [])
    assert not isinstance(exc.value, LLMUnavailableError) and client.calls == ["primary"]
