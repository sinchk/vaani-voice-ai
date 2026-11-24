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
    GeminiLLM("primary", max_retries=3, fallback_models=[]).generate("sys", [], [])
    assert client.calls == ["primary"] * 3


def test_falls_back_to_secondary_model(monkeypatch):
    client = install(monkeypatch, lambda m, n: api_error(503) if m == "primary" else FakeResponse())
    GeminiLLM("primary", max_retries=1, fallback_models=["backup"]).generate("sys", [], [])
    assert client.calls == ["primary", "primary", "backup"]


def test_raises_unavailable_when_everything_is_down(monkeypatch):
    install(monkeypatch, lambda m, n: api_error(503))
    with pytest.raises(LLMUnavailableError):
        GeminiLLM("primary", max_retries=1, fallback_models=["backup"]).generate("sys", [], [])


def test_client_errors_are_not_retried(monkeypatch):
    client = install(monkeypatch, lambda m, n: api_error(400))
    with pytest.raises(LLMError) as exc:
        GeminiLLM("primary", max_retries=3, fallback_models=["backup"]).generate("sys", [], [])
    assert not isinstance(exc.value, LLMUnavailableError) and client.calls == ["primary"]


def test_daily_quota_skips_retries_and_uses_fallback(monkeypatch):
    quota = errors.APIError(429, {"error": {"code": 429, "message": "Requests per day limit exceeded", "status": "X"}})
    client = install(monkeypatch, lambda m, n: quota if m == "primary" else FakeResponse())
    GeminiLLM("primary", max_retries=3, fallback_models=["backup"]).generate("sys", [], [])
    assert client.calls == ["primary", "backup"]


def test_fallback_chain_tries_each_model_in_order(monkeypatch):
    client = install(monkeypatch, lambda m, n: FakeResponse() if m == "third" else api_error(503))
    GeminiLLM("first", max_retries=0, fallback_models=["second", "third"]).generate("sys", [], [])
    assert client.calls == ["first", "second", "second", "third"]


def test_gemini_asr_transcribes_and_detects_language(monkeypatch, tmp_path):
    from vaani.speech import asr as asr_module

    class Resp:
        text = " ನನ್ನ ಆರ್ಡರ್ 1043 ಎಲ್ಲಿದೆ? "

    seen = {}

    def fake_generate(model, contents, config, retries):
        seen["model"] = model
        return Resp()

    monkeypatch.setattr("vaani.agent.llm.generate_with_retry", fake_generate)
    clip = tmp_path / "a.mp3"
    clip.write_bytes(b"fake")
    result = asr_module.GeminiASR(models=["m1"]).transcribe(str(clip))
    assert result.text == "ನನ್ನ ಆರ್ಡರ್ 1043 ಎಲ್ಲಿದೆ?" and result.language == "kn" and seen["model"] == "m1"
