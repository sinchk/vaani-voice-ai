"""Small script-based language helpers (no model needed)."""

from __future__ import annotations

_KANNADA_START, _KANNADA_END = 0x0C80, 0x0CFF


def kannada_ratio(text: str) -> float:
    """Fraction of alphabetic characters written in Kannada script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(_KANNADA_START <= ord(c) <= _KANNADA_END for c in letters) / len(letters)


def detect_language(text: str, threshold: float = 0.25) -> str:
    """Return 'kn' if a meaningful share of the text is Kannada script, else 'en'.

    A low threshold means code-switched text that is substantially Kannada is
    voiced in Kannada, where gTTS also pronounces embedded English acceptably.
    """
    return "kn" if kannada_ratio(text) >= threshold else "en"
