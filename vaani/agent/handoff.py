"""Deterministic human-handoff detection.

The LLM can also escalate via the ``request_human_handoff`` tool. This keyword
check is a safety net so an explicit request for a person is *always* honoured,
even if the model fails to call the tool or the LLM is unavailable.
"""

from __future__ import annotations

import re

_PATTERNS = [
    # English: "human", "real person", "talk to an agent", "representative", "supervisor", "escalate"
    r"\b(human|real person|live agent|agent|representative|supervisor|manager|escalat\w*)\b",
    # Kannada: ಏಜೆಂಟ್ (agent), ಮಾನವ/ಮನುಷ್ಯ (human), ವ್ಯಕ್ತಿ (person), ಪ್ರತಿನಿಧಿ (representative)
    r"ಏಜೆಂಟ್|ಮಾನವ|ಮನುಷ್ಯ|ವ್ಯಕ್ತಿಯ|ಪ್ರತಿನಿಧಿ|ಮ್ಯಾನೇಜರ್",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def wants_human(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)
