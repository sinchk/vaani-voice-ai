import pytest

from tests.conftest import FakeLLM, FakeRetriever, call_turn, make_agent, text_turn
from vaani.agent.llm import LLMUnavailableError


def user_text(content) -> str:
    return "".join(p.text or "" for p in content.parts)


def test_plain_answer_uses_retrieved_context(registry, return_chunk):
    llm = FakeLLM(text_turn("You can return items within 30 days."))
    agent = make_agent(registry, llm, FakeRetriever([return_chunk]))

    result = agent.chat("s1", "What is your return policy?")

    assert result.response == "You can return items within 30 days."
    assert result.sources == [return_chunk]
    assert result.confidence == 0.82
    assert result.tool_calls == [] and not result.handoff
    sent = user_text(llm.calls[0]["contents"][-1])
    assert "within 30 days" in sent and sent.startswith("What is your return policy?")


def test_no_context_means_no_confidence_claim(registry):
    agent = make_agent(registry, FakeLLM(text_turn("Hi!")))
    assert agent.chat("s1", "hello").confidence is None


def test_tool_call_round_trip(registry):
    llm = FakeLLM(call_turn("cancel_order", order_id="1043"), text_turn("Order 1043 is cancelled."))
    agent = make_agent(registry, llm)

    result = agent.chat("s1", "please cancel order 1043")

    assert result.response == "Order 1043 is cancelled."
    assert [c["name"] for c in result.tool_calls] == ["cancel_order"]
    assert result.tool_calls[0]["result"]["ok"] is True
    assert registry.orders.get_order_status("1043")["order"]["status"] == "Cancelled"
    # The second LLM call must carry the model's call and the tool's response.
    second = llm.calls[1]["contents"]
    assert second[-2].parts[0].function_call.name == "cancel_order"
    assert second[-1].parts[0].function_response.response["ok"] is True


def test_tool_failure_is_passed_back_to_the_model(registry):
    llm = FakeLLM(call_turn("cancel_order", order_id="1042"), text_turn("Sorry, it already shipped."))
    result = make_agent(registry, llm).chat("s1", "cancel 1042")
    assert result.tool_calls[0]["result"]["ok"] is False
    assert result.response == "Sorry, it already shipped."


def test_history_is_sent_on_later_turns(registry):
    llm = FakeLLM(text_turn("Which order?"), text_turn("Order 1042 has shipped."))
    agent = make_agent(registry, llm)
    agent.chat("s1", "where is my order")
    agent.chat("s1", "1042")

    roles_and_text = [(c.role, user_text(c)) for c in llm.calls[1]["contents"]]
    assert roles_and_text == [("user", "where is my order"), ("model", "Which order?"), ("user", "1042")]


def test_sessions_are_isolated(registry):
    llm = FakeLLM(text_turn("a"), text_turn("b"))
    agent = make_agent(registry, llm)
    agent.chat("s1", "first")
    agent.chat("s2", "second")
    assert len(llm.calls[1]["contents"]) == 1


def test_slots_are_remembered_from_tool_args(registry):
    agent = make_agent(registry, FakeLLM(call_turn("get_order_status", order_id="1042"), text_turn("ok")))
    agent.chat("s1", "status of 1042")
    assert agent.memory.get_slots("s1")["last_order_id"] == "1042"


def test_handoff_via_model_tool(registry):
    llm = FakeLLM(call_turn("request_human_handoff", reason="refund dispute"), text_turn("Connecting you."))
    result = make_agent(registry, llm).chat("s1", "this is unacceptable")
    assert result.handoff
    assert result.handoff_summary["reason"] == "refund dispute"
    assert result.handoff_summary["recent_messages"][-1]["text"] == "Connecting you."


@pytest.mark.parametrize("message", ["I want to talk to a human agent", "ನನಗೆ ಒಬ್ಬ ಏಜೆಂಟ್ ಬೇಕು"])
def test_handoff_keyword_safety_net_even_if_llm_is_down(registry, message):
    class DownLLM:
        def generate(self, *a, **k):
            raise LLMUnavailableError("down")

    result = make_agent(registry, DownLLM()).chat("s1", message)
    assert result.handoff and result.response
    assert result.handoff_summary["reason"] == "Customer asked for a human agent."


def test_llm_outage_propagates_for_normal_requests(registry):
    class DownLLM:
        def generate(self, *a, **k):
            raise LLMUnavailableError("down")

    with pytest.raises(LLMUnavailableError):
        make_agent(registry, DownLLM()).chat("s1", "hello")


def test_retrieval_failure_degrades_gracefully(registry):
    agent = make_agent(registry, FakeLLM(text_turn("Hello!")), FakeRetriever(error=RuntimeError("chroma down")))
    result = agent.chat("s1", "hi")
    assert result.response == "Hello!" and result.sources == []


def test_runaway_tool_loop_is_capped(registry):
    llm = FakeLLM(*[call_turn("get_order_status", order_id="1042") for _ in range(10)])
    agent = make_agent(registry, llm)
    result = agent.chat("s1", "loop forever")
    assert len(llm.calls) == agent.max_tool_rounds
    assert result.response  # falls back to an apology rather than returning empty text
