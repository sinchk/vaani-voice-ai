import pytest
from vaani.app.agents.agent import VaaniAgent

def test_agent_initialization():
    agent = VaaniAgent()
    assert agent is not None

def test_agent_chat_response():
    # This requires a valid API key in .env or environment
    # Skipping actual API call in unit test for now or mocking it
    pass
