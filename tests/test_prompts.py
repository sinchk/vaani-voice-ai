from datetime import date

from vaani.agent.prompts import build_system_prompt, build_user_message


def test_system_prompt_carries_todays_date_for_relative_dates():
    prompt = build_system_prompt(date(2026, 12, 24))
    assert "2026-12-24" in prompt and "Thursday" in prompt


def test_system_prompt_forces_tool_verification_for_cancellations():
    assert "always call `cancel_order`" in build_system_prompt()


def test_user_message_without_context_is_unchanged():
    assert build_user_message("hello", []) == "hello"


def test_user_message_appends_context_without_altering_the_question():
    msg = build_user_message("return policy?", ["30 days"])
    assert msg.startswith("return policy?") and "- 30 days" in msg
