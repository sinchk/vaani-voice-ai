# Vaani — Multilingual Agentic Voice AI

Vaani is a production-ready multilingual agentic voice AI supporting English, Kannada, and code-switched speech. It integrates RAG for knowledge retrieval, tool calling for business logic, and a streamlined frontend.

## Features
- **Multilingual ASR & TTS**: English and Kannada support using Whisper and gTTS.
- **Agentic Core**: Powered by Google Gemini with tool calling capabilities.
- **RAG Knowledge Base**: Persistent vector store using ChromaDB and multilingual embeddings.
- **Interactive UI**: Streamlit-based frontend for voice and chat interactions.
- **Evaluation**: Automated benchmarks for retrieval, tools, and groundedness.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` with your `GEMINI_API_KEY`.
3. Run the backend: `uvicorn app.main:app --reload`
4. Run the frontend: `streamlit run frontend/app.py`
