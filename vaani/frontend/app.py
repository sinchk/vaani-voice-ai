import streamlit as st
import requests
import os
import uuid

st.set_page_config(page_title="Vaani — Multilingual Voice AI", layout="wide")

st.title("Vaani — Multilingual Agentic Voice AI")
st.markdown("Interact with Vaani using English, Kannada, or mixed speech.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar for session info and tool logs
with st.sidebar:
    st.header("Session Settings")
    st.write(f"Session ID: `{st.session_state.session_id}`")
    if st.button("Reset Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    st.header("Execution Logs")
    # Placeholder for tool logs
    st.info("No tool executions yet.")

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat")
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        response = requests.post("http://localhost:8000/api/chat", json={
            "session_id": st.session_state.session_id,
            "message": user_input
        }).json()
        
        with st.chat_message("assistant"):
            st.write(response["response"])
            if response.get("sources"):
                with st.expander("Sources"):
                    for src in response["sources"]:
                        st.write(f"- {src['content']} (Confidence: {src['confidence']})")
            
            if response.get("handoff"):
                st.warning("⚠️ Human Handoff Triggered")

with col2:
    st.subheader("Voice Interaction")
    
    # 1. New st.audio_input for direct recording
    recorded_audio = st.audio_input("Record your voice (English / Kannada)")
    
    # 2. Existing file uploader for compatibility
    uploaded_file = st.file_uploader("Or upload an Audio file", type=["wav", "mp3"])
    
    active_audio = recorded_audio if recorded_audio else uploaded_file
    
    if active_audio:
        st.audio(active_audio)
        if st.button("Process Voice"):
            with st.spinner("Processing speech..."):
                res = requests.post(
                    "http://localhost:8000/api/voice",
                    data={"session_id": st.session_state.session_id},
                    files={"file": (active_audio.name if hasattr(active_audio, 'name') else "input.wav", active_audio.getvalue(), "audio/wav")}
                ).json()
                
                st.write(f"**Transcript:** {res['transcript']}")
                st.write(f"**Response:** {res['response']}")
                
                if res.get("audio_url") and os.path.exists(res["audio_url"]):
                    st.audio(res["audio_url"])

                # Display Execution Details in Sidebar
                with st.sidebar:
                    st.header("Last Interaction")
                    st.metric("Confidence", f"{res.get('confidence', 0.0) * 100}%")
                    if res.get("tool_calls"):
                        st.subheader("Tools Called")
                        for tc in res["tool_calls"]:
                            st.code(f"{tc['name']}({tc['args']})")
                    
                    if res.get("sources"):
                        st.subheader("Knowledge Sources")
                        for src in res["sources"]:
                            st.caption(f"- {src['source']} (Conf: {src['confidence']})")
                    
                    if res.get("handoff"):
                        st.error("Human Handoff Triggered")
                        st.json(res.get("handoff_summary", {}))

st.divider()
st.caption("Built with Gemini, ChromaDB, and Streamlit.")
