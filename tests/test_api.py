import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeLLM, FakeRetriever, call_turn, make_agent, text_turn
from vaani.agent.llm import LLMUnavailableError
from vaani.api.app import create_app
from vaani.api.deps import get_agent, get_asr, get_tools, get_tts
from vaani.speech.asr import Transcript
from vaani.speech.tts import TTSProcessor


class FakeASR:
    def __init__(self, text="where is order 1042"):
        self.text = text

    def transcribe(self, path):
        return Transcript(self.text, "en", 0.99)


class FakeTTS(TTSProcessor):
    def synthesize(self, text, lang=None):
        name = "response_" + "0" * 8 + "-0000-0000-0000-" + "0" * 12 + ".mp3"
        (self.output_dir / name).write_bytes(b"ID3fake")
        self.last = (text, lang)
        return name


@pytest.fixture
def make_client(registry, tmp_path):
    def _make(llm, asr=None):
        app = create_app()
        app.dependency_overrides[get_agent] = lambda: make_agent(registry, llm, FakeRetriever())
        app.dependency_overrides[get_tools] = lambda: registry
        app.dependency_overrides[get_asr] = lambda: asr or FakeASR()
        tts = FakeTTS(output_dir=tmp_path / "audio")
        app.dependency_overrides[get_tts] = lambda: tts
        client = TestClient(app, raise_server_exceptions=False)
        client.tts = tts
        return client

    return _make


def test_health(make_client):
    r = make_client(FakeLLM()).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_chat_with_tool_call(make_client):
    llm = FakeLLM(call_turn("get_order_status", order_id="1042"), text_turn("Order 1042 has shipped."))
    r = make_client(llm).post("/api/chat", json={"session_id": "s", "message": "status of 1042"})
    body = r.json()
    assert r.status_code == 200
    assert body["response"] == "Order 1042 has shipped."
    assert body["tool_calls"][0]["name"] == "get_order_status"
    assert body["confidence"] is None and body["handoff"] is False


@pytest.mark.parametrize(
    "payload", [{"session_id": "s", "message": ""}, {"session_id": "", "message": "hi"}, {"message": "hi"}]
)
def test_chat_validation(make_client, payload):
    assert make_client(FakeLLM()).post("/api/chat", json=payload).status_code == 422


def test_chat_maps_llm_outage_to_503(make_client):
    class Down:
        def generate(self, *a, **k):
            raise LLMUnavailableError("down")

    r = make_client(Down()).post("/api/chat", json={"session_id": "s", "message": "hello"})
    assert r.status_code == 503 and "busy" in r.json()["detail"]


def test_voice_round_trip(make_client):
    client = make_client(FakeLLM(text_turn("It has shipped.")))
    r = client.post("/api/voice", data={"session_id": "s"}, files={"file": ("a.wav", b"RIFFxxxx", "audio/wav")})
    body = r.json()
    assert r.status_code == 200, body
    assert body["transcript"] == "where is order 1042" and body["response"] == "It has shipped."
    assert body["audio_url"].startswith("/api/audio/response_")
    assert client.get(body["audio_url"]).content == b"ID3fake"  # audio is served over HTTP, not a server path


def test_voice_rejects_bad_type_and_silence(make_client):
    client = make_client(FakeLLM(), asr=FakeASR(text=""))
    assert client.post("/api/voice", data={"session_id": "s"}, files={"file": ("a.exe", b"x")}).status_code == 415
    assert client.post("/api/voice", data={"session_id": "s"}, files={"file": ("a.wav", b"x")}).status_code == 422


def test_audio_endpoint_blocks_path_traversal_and_unknown_files(make_client):
    client = make_client(FakeLLM())
    assert client.get("/api/audio/..%2F..%2Fsecret.txt").status_code in (404, 422)
    assert client.get("/api/audio/response_nope.mp3").status_code == 404


def test_tool_endpoints(make_client):
    client = make_client(FakeLLM())
    assert client.get("/api/tools/orders/1042").json()["order"]["status"] == "Shipped"
    assert client.post("/api/tools/orders/1043/cancel").json()["ok"] is True
    assert client.get("/api/tools/appointments", params={"date": "2031-01-01"}).json()["ok"] is True
