from vaani.agent.handoff import wants_human
from vaani.agent.memory import SessionMemory
from vaani.rag.ingest import chunk_text, load_chunks
from vaani.rag.retriever import KBRetriever
from vaani.text_utils import detect_language


class TestChunking:
    def test_short_paragraph_is_one_chunk(self):
        assert chunk_text("One sentence. Two sentences.") == ["One sentence. Two sentences."]

    def test_long_text_splits_on_sentence_boundaries(self):
        text = " ".join(f"This is sentence number {i}." for i in range(30))
        chunks = chunk_text(text, max_chars=100)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)
        assert all(c.endswith(".") for c in chunks)

    def test_blank_input(self):
        assert chunk_text("  \n\n ") == []

    def test_real_knowledge_base_loads_with_stable_ids(self):
        from vaani.config import get_settings

        items = load_chunks(get_settings().knowledge_base_dir)
        assert {i["source"] for i in items} >= {"return_policy.txt", "shipping_policy.txt", "cancellation_policy.txt"}
        assert len({i["id"] for i in items}) == len(items)


class FakeCollection:
    def __init__(self, distances):
        self.distances = distances

    def count(self):
        return 3

    def query(self, **_):
        n = len(self.distances)
        return {
            "documents": [[f"doc{i}" for i in range(n)]],
            "metadatas": [[{"source": f"f{i}.txt"} for i in range(n)]],
            "distances": [self.distances],
        }


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0]


def test_retriever_converts_cosine_distance_and_applies_floor(monkeypatch):
    monkeypatch.setattr("vaani.rag.retriever.get_collection", lambda: FakeCollection([0.1, 0.4, 0.7]))
    chunks = KBRetriever(embedder=FakeEmbedder(), min_score=0.55, margin=1.0).retrieve("q")
    assert [(c.source, c.confidence) for c in chunks] == [("f0.txt", 0.9), ("f1.txt", 0.6)]


def test_retriever_drops_chunks_far_below_the_best_match(monkeypatch):
    monkeypatch.setattr("vaani.rag.retriever.get_collection", lambda: FakeCollection([0.1, 0.12, 0.4]))
    chunks = KBRetriever(embedder=FakeEmbedder(), min_score=0.55, margin=0.05).retrieve("q")
    assert [c.source for c in chunks] == ["f0.txt", "f1.txt"]


def test_retriever_returns_nothing_for_off_topic_queries(monkeypatch):
    monkeypatch.setattr("vaani.rag.retriever.get_collection", lambda: FakeCollection([0.5, 0.6]))
    assert KBRetriever(embedder=FakeEmbedder(), min_score=0.62).retrieve("capital of France") == []


class TestHandoffDetection:
    def test_positive(self):
        for text in ["I want a human", "get me a representative", "transfer to an agent", "ನನಗೆ ಏಜೆಂಟ್ ಬೇಕು"]:
            assert wants_human(text), text

    def test_negative(self):
        for text in [
            "Where is my order 1042?",
            "What is the return policy?",
            "ನನ್ನ ಆರ್ಡರ್ ಎಲ್ಲಿದೆ?",
            "my humanoid robot arrived broken",
        ]:
            assert not wants_human(text), text


class TestLanguage:
    def test_english(self):
        assert detect_language("Your order has shipped.") == "en"

    def test_kannada(self):
        assert detect_language("ನಿಮ್ಮ ಆರ್ಡರ್ ಕಳುಹಿಸಲಾಗಿದೆ.") == "kn"

    def test_code_switched_leaning_kannada(self):
        assert detect_language("ನಿಮ್ಮ order 1042 ಕಳುಹಿಸಲಾಗಿದೆ shipped") == "kn"

    def test_empty(self):
        assert detect_language("") == "en"


class TestMemory:
    def test_history_window_is_bounded_and_starts_on_user_turn(self):
        m = SessionMemory(max_messages=4)
        for i in range(5):
            m.add_message("s", "user", f"u{i}")
            m.add_message("s", "model", f"m{i}")
        msgs = m.get_messages("s")
        assert len(msgs) <= 4 and msgs[0]["role"] == "user" and msgs[-1]["text"] == "m4"

    def test_idle_sessions_are_evicted(self):
        m = SessionMemory(max_sessions=2)
        for s in ("a", "b", "c"):
            m.add_message(s, "user", "x")
        assert m.get_messages("a") == []  # evicted, recreated empty

    def test_get_messages_returns_a_copy(self):
        m = SessionMemory()
        m.add_message("s", "user", "x")
        m.get_messages("s").clear()
        assert len(m.get_messages("s")) == 1
