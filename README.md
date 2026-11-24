# Vaani â€” Multilingual Agentic Voice AI

Vaani is a customer-support assistant that you can **talk or type to in English, Kannada (à²•à²¨à³à²¨à²¡), or a mix of both**. It answers policy questions from a knowledge base (RAG), performs real actions through tools (order lookup/cancellation, appointment booking), remembers the conversation, and hands off to a human when it should.

```
 voice â”€â–º Whisper (ASR) â”€â”
                         â–¼
 text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Agent â”€â”€â–º Gemini (function calling) â”€â”€â–º tools: orders, appointments, handoff
                         â–²  â””â”€â–º reply â”€â–º gTTS (TTS) â”€â–º voice
                         â”‚
              ChromaDB â—„â”€â”˜  multilingual embeddings (Kannada query â†’ English policy)
```

## Highlights

- **Real tool calling.** Gemini decides which tool to call and with what arguments (function calling), so "cancel it", "à²¨à²¨à³à²¨ à²†à²°à³à²¡à²°à³ à²Žà²²à³à²²à²¿à²¦à³†?" or "book me in tomorrow at 10" all work without keyword rules. Each tool call and its result is returned to the UI.
- **Grounded answers.** Policy questions are answered from retrieved knowledge-base chunks. Off-topic questions retrieve nothing, and the assistant is instructed to say so rather than invent an answer.
- **Honest confidence.** The `confidence` field is the cosine similarity of the best retrieved chunk, or `null` when no knowledge was used. It is never a made-up default.
- **Conversation memory.** Per-session history is sent to the model, so follow-ups ("1042" after "where is my order?") work. Memory is bounded.
- **Human handoff, two ways.** The model can call `request_human_handoff`, and a deterministic English/Kannada keyword check guarantees an explicit request for a person is always honoured â€” even if the LLM is down. The handoff includes a summary (reason, slots, recent messages, tools used).
- **Resilient.** Retries with backoff on 429/5xx, automatic fallback-model chain, graceful degradation if retrieval fails, and clear HTTP errors (`503` busy, `502` failed) instead of fake replies.
- **Two speech-recognition backends.** Whisper runs fully offline (good for English); `VAANI_ASR_BACKEND=gemini` uses Gemini's native audio understanding, which is what you want for Kannada (see [evaluation](docs/evaluation.md)).
- **Voice that works over HTTP.** `/api/voice` takes audio, returns transcript + reply + a URL to the spoken reply; the TTS language follows the script of the reply.
- **Demo data never dirties git.** Seed data in `data/` is copied to `.state/` on first use; mutations happen there.

## Quick start

Requires Python 3.10+ and a [Gemini API key](https://aistudio.google.com/apikey). `ffmpeg` is needed for non-WAV audio.

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then set GEMINI_API_KEY

uvicorn vaani.api.app:app --reload                     # API  â†’ http://localhost:8000/docs
streamlit run frontend/app.py                          # UI   â†’ http://localhost:8501
```

The knowledge base is embedded automatically on first start (or run `python -m vaani.rag.ingest --reset`). Whisper weights download on the first voice request.

### Docker

```bash
cp .env.example .env   # set GEMINI_API_KEY
docker compose up --build
```

API on `:8000`, UI on `:8501`; state, vector store and model weights persist in named volumes.

## Try it

| Say | What happens |
|---|---|
| "What is your return policy?" | RAG â†’ answer citing the 30-day policy, with sources |
| "Where is my order 1042?" | `get_order_status("1042")` |
| "Cancel order 1043" | `cancel_order("1043")` (order 1042 is shipped, so it is refused with an explanation) |
| "Book an appointment for Sinchana tomorrow at 10 AM" | model resolves the date â†’ `book_appointment(...)` |
| "à²¨à²¨à³à²¨ à²†à²°à³à²¡à²°à³ 1043 à²¸à³à²¥à²¿à²¤à²¿ à²à²¨à³?" | Kannada in, Kannada out, same tool |
| "I want to talk to a human" | handoff + summary |

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | `{session_id, message}` â†’ reply, `sources`, `tool_calls`, `confidence`, `handoff`, `handoff_summary` |
| `POST /api/voice` | multipart `session_id` + `file` â†’ everything above plus `transcript`, `detected_language`, `audio_url` |
| `GET /api/audio/{name}` | the synthesized reply (mp3; auto-expires) |
| `GET/POST /api/tools/...` | call the business tools directly |
| `GET /api/health` | liveness, model, whether a key is configured |

Interactive docs: `/docs`.

Full write-up (architecture, request flows, results, limitations, where it fits best): [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

## Project layout

```
vaani/
  config.py            all settings (env-driven, paths anchored to the repo)
  agent/               agent loop, Gemini wrapper (retry/fallback), memory, prompts, handoff
  tools/               orders, appointments, JSON store, registry (schemas + implementations)
  rag/                 embeddings, ingestion/chunking, retriever
  speech/              faster-whisper ASR, gTTS TTS
  api/                 FastAPI app, routes, schemas, dependencies
  evaluation/          end-to-end benchmark
frontend/app.py        Streamlit UI
data/                  seed orders/appointments + knowledge-base documents
tests/                 unit + API tests (no network needed)
```

To add a tool, write the function and add one `ToolSpec` in `vaani/tools/registry.py`; the schema Gemini sees and the code that runs it live together. To extend the knowledge base, drop a `.txt` file into `data/knowledge_base/` and run `python -m vaani.rag.ingest`.

## Configuration

Everything is optional except `GEMINI_API_KEY`; see [`.env.example`](.env.example). Notable knobs: `VAANI_CHAT_MODEL`, `VAANI_FALLBACK_MODELS`, `VAANI_ASR_BACKEND` (`gemini` for Kannada voice), `VAANI_WHISPER_MODEL`, `VAANI_RAG_MIN_SCORE`.

## Results

Measured on a live single-pass run of all 19 cases (details and per-case table in [docs/evaluation.md](docs/evaluation.md)):

| Metric | Result |
|---|---|
| Tool accuracy (name + arguments) | 100% (7/7) |
| RAG recall@3 | 100% (6/6, incl. Kannada) |
| Answer-fact coverage (EN) | 100% (6/6) |
| Handoff accuracy (incl. negatives) | 100% (3/3) |
| No-tool accuracy | 100% (3/3) |
| Latency p50 / p95 | 1.9 s / 8.2 s on `gemini-3.5-flash-lite` (a tool-calling turn makes two LLM calls) |

The suite is small (19 cases), so read these as a regression check, not a benchmark of general quality.

## Testing and evaluation

```bash
pip install -r requirements-dev.txt
pytest          # 81 tests, fully offline (LLM, retriever, ASR and TTS are faked)
ruff check .
```

The benchmark runs 19 realistic cases (English + Kannada: RAG, tools, no-tool, handoff, multi-turn) against a live server and reports tool accuracy, RAG recall@3, answer-fact coverage, handoff accuracy and latency percentiles:

```bash
VAANI_STATE_DIR=.state-eval uvicorn vaani.api.app:app &   # separate state: the benchmark books appointments
python -m vaani.evaluation.benchmark --report docs/evaluation.md
```

## Limitations

- The Gemini **free tier allows only ~20 generate requests per model per day**; the fallback chain stretches this, but for real use enable billing on your key.
- The API has **no authentication or rate limiting** (anyone who can reach it can cancel demo orders). Put it behind your own auth/gateway before exposing it.
- Sessions and memory live in process (restarting clears them); use Redis or a database for multi-instance deployments.
- gTTS needs internet access. Whisper on CPU is poor at Kannada even at `medium` (and slow); use `VAANI_ASR_BACKEND=gemini` for Kannada voice.
- Orders and appointments are JSON files, a stand-in for a real backend; the `OrderService` / `AppointmentService` interfaces are where a database or API would plug in.
- Retrieval scores from Gemini embeddings are compressed (unrelated text still scores about 0.55), so `VAANI_RAG_MIN_SCORE` was tuned empirically (0.68) on this small knowledge base and should be re-tuned if you add many documents.
