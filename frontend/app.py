"""Streamlit UI for Vaani.  Run:  streamlit run frontend/app.py"""

from __future__ import annotations

import hashlib
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("VAANI_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120

st.set_page_config(page_title="Vaani - Multilingual Voice AI", page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------------------- API
def call_api(method: str, path: str, **kwargs):
    """Call the backend; show a friendly error and return None on failure."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError:
        st.error(f"Cannot reach the Vaani API at {API_URL}. Start it with `uvicorn vaani.api.app:app`.")
        return None
    except requests.Timeout:
        st.error("The request timed out. Please try again.")
        return None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Request failed ({resp.status_code}): {detail}")
        return None
    return resp


@st.cache_data(ttl=10, show_spinner=False)
def backend_health() -> dict | None:
    try:
        return requests.get(f"{API_URL}/api/health", timeout=3).json()
    except (requests.RequestException, ValueError):
        return None


def fetch_audio(audio_url: str | None) -> bytes | None:
    if not audio_url:
        return None
    resp = call_api("GET", audio_url)
    return resp.content if resp is not None else None


# ----------------------------------------------------------------------------- state
def reset_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.last_voice_hash = None


if "session_id" not in st.session_state:
    reset_session()


def add_assistant(data: dict, audio: bytes | None = None) -> None:
    st.session_state.messages.append({"role": "assistant", "content": data["response"], "meta": data, "audio": audio})


# ----------------------------------------------------------------------------- rendering
def render_meta(meta: dict) -> None:
    if meta.get("handoff"):
        st.warning("Handing over to a human agent")
        with st.expander("Handoff summary"):
            st.json(meta.get("handoff_summary", {}))
    if meta.get("tool_calls"):
        with st.expander(f"Tools used ({len(meta['tool_calls'])})"):
            for call in meta["tool_calls"]:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                st.code(f"{call['name']}({args})", language="python")
                st.json(call["result"], expanded=False)
    if meta.get("sources"):
        with st.expander(f"Knowledge sources ({len(meta['sources'])})"):
            for src in meta["sources"]:
                st.markdown(f"**{src['source']}** · similarity {src['confidence']:.2f}")
                st.caption(src["content"])


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mpeg")
        if msg.get("meta"):
            render_meta(msg["meta"])


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Vaani")
    health = backend_health()
    if health is None:
        st.error("Backend offline")
    elif not health.get("api_key_configured"):
        st.warning("Backend is up but GEMINI_API_KEY is not set")
    else:
        st.success(f"Backend online · {health['model']}")

    st.caption(f"Session `{st.session_state.session_id[:8]}`")
    if st.button("New conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.subheader("Try asking")
    st.markdown(
        "- What is your return policy?\n"
        "- Where is my order 1042?\n"
        "- Cancel order 1043\n"
        "- Book an appointment for me tomorrow at 10 AM\n"
        "- ನನ್ನ ಆರ್ಡರ್ 1043 ಸ್ಥಿತಿ ಏನು?\n"
        "- I want to talk to a human"
    )

    last = next((m["meta"] for m in reversed(st.session_state.messages) if m.get("meta")), None)
    if last:
        st.divider()
        st.subheader("Last reply")
        conf = last.get("confidence")
        st.metric(
            "Knowledge match",
            f"{conf:.0%}" if conf is not None else "-",
            help="Similarity of the best knowledge-base chunk used. Empty when the answer needed none.",
        )
        st.caption("Tools: " + (", ".join(c["name"] for c in last["tool_calls"]) or "none"))

# ----------------------------------------------------------------------------- main
st.title("Vaani")
st.caption("Multilingual agentic voice assistant · English · ಕನ್ನಡ · mixed")

for message in st.session_state.messages:
    render_message(message)

with st.expander("🎙️ Speak instead", expanded=not st.session_state.messages):
    recorded = st.audio_input("Record your message")
    uploaded = st.file_uploader("...or upload audio", type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])
    clip = recorded or uploaded
    if clip is not None:
        audio_bytes = clip.getvalue()
        digest = hashlib.sha1(audio_bytes).hexdigest()
        if digest != st.session_state.last_voice_hash and st.button("Send voice message", type="primary"):
            st.session_state.last_voice_hash = digest  # never submit the same clip twice
            with st.spinner("Listening..."):
                resp = call_api(
                    "POST",
                    "/api/voice",
                    data={"session_id": st.session_state.session_id},
                    files={"file": (clip.name or "recording.wav", audio_bytes, clip.type or "audio/wav")},
                )
            if resp is not None:
                data = resp.json()
                st.session_state.messages.append({"role": "user", "content": f"🎙️ {data['transcript']}"})
                add_assistant(data, fetch_audio(data.get("audio_url")))
                st.rerun()

if prompt := st.chat_input("Type your message (English or ಕನ್ನಡ)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(st.session_state.messages[-1])
    with st.spinner("Thinking..."):
        resp = call_api("POST", "/api/chat", json={"session_id": st.session_state.session_id, "message": prompt})
    if resp is not None:
        add_assistant(resp.json())
        st.rerun()
