# Vaani — Project Report

**Multilingual (English / Kannada) agentic voice AI for customer support**
Version 0.2.0 · Repository: `sinchk/vaani-voice-ai`

---

## 1. Summary

Vaani is a customer-support assistant that a person can **talk or type to in English, Kannada (ಕನ್ನಡ), or a mix of both**. It:

1. **Understands** speech (ASR) or text,
2. **Decides** what to do using a large language model with *function calling*,
3. **Looks things up** in a policy knowledge base (RAG) and **takes real actions** through tools (check/cancel an order, check/book an appointment),
4. **Remembers** the conversation, and **escalates to a human** with a summary when it should,
5. **Replies** in text and synthesized speech, in the customer's language.

The project started as a working-looking prototype whose "AI" was largely regular expressions and hard-coded values. It was rebuilt into a tested, configurable, deployable system and validated against the live Gemini API.

---

## 2. Technology stack

| Layer | Technology | Why it was chosen |
|---|---|---|
| Language model | **Google Gemini** (`gemini-3.8-flash` default, fallback chain) via the `google-genai` SDK | Strong multilingual (Kannada) ability, native function calling, native audio input |
| Embeddings | **`gemini-embedding-001`** | Multilingual: a Kannada question lands near an English policy document |
| Vector store | **ChromaDB** (persistent, cosine space) | Zero-ops, embedded, simple |
| Speech-to-text | **faster-whisper** (offline) *or* **Gemini audio** (selectable) | Whisper = private/offline, good for English; Gemini = accurate Kannada |
| Text-to-speech | **gTTS** (English + Kannada) | Free, supports Kannada |
| Backend | **FastAPI** + Uvicorn, **Pydantic v2** | Typed contracts, automatic OpenAPI docs at `/docs` |
| Frontend | **Streamlit** | Fast to build chat + audio recorder UI |
| Packaging | **Docker**, **docker-compose** | One-command run: API + UI, health-gated, persistent volumes |
| Quality | **pytest** (81 tests), **ruff**, **GitHub Actions** CI (Python 3.10 & 3.12) | Offline, deterministic tests; lint + tests on every push |

---

## 3. Architecture

```
                      ┌────────────── Streamlit UI (frontend/app.py) ──────────────┐
                      │  chat box · mic recorder / upload · sources · tool log     │
                      └───────────────┬───────────────────────────┬────────────────┘
                              POST /api/chat              POST /api/voice
                                      │                           │
                                      │                 ┌─────────▼─────────┐
                                      │                 │  ASR              │
                                      │                 │  Whisper | Gemini │
                                      │                 └─────────┬─────────┘
                              ┌───────▼───────────────────────────▼───────┐
                              │                VaaniAgent                 │
                              │  1. retrieve KB context (RAG)             │
                              │  2. build prompt = history + question     │
                              │     + context + today's date              │
                              │  3. LLM ⇄ tool loop (max 4 rounds)        │
                              │  4. handoff check (model tool + keywords) │
                              │  5. save to session memory                │
                              └──┬──────────┬───────────────┬─────────────┘
                                 │          │               │
                       ┌─────────▼──┐  ┌────▼──────┐  ┌─────▼───────────────┐
                       │ KBRetriever│  │ GeminiLLM │  │   ToolRegistry      │
                       │ Chroma +   │  │ retry +   │  │ get_order_status    │
                       │ embeddings │  │ fallback  │  │ cancel_order        │
                       └────────────┘  │ chain     │  │ check_appointment   │
                                       └───────────┘  │ book_appointment    │
                                                      │ request_human_handoff│
                                                      └─────────────────────┘
                                 reply ─► TTS (gTTS) ─► mp3 served at /api/audio/{name}
```

### Package layout

```
vaani/
  config.py          env-driven settings; every path anchored to the repo root
  gemini.py          one shared Gemini client
  text_utils.py      script-based language detection (Kannada vs English)
  agent/             agent loop · LLM wrapper (retry/fallback) · memory · prompts · handoff
  tools/             order & appointment services · JSON store · tool registry
  rag/               embeddings · chunking + ingestion · retriever
  speech/            Whisper & Gemini ASR · gTTS
  api/               FastAPI app, routes, schemas, dependency providers
  evaluation/        end-to-end benchmark
frontend/app.py      Streamlit UI
data/                seed orders/appointments + knowledge-base documents
tests/               81 offline tests
docs/                this report + evaluation results
```

---

## 4. How it works — step by step

### 4.1 A text turn: *"Cancel order 1043"*

1. **API** validates `{session_id, message}` (Pydantic: non-empty, length-capped) and hands it to the agent.
2. **Retrieval.** The question is embedded (`RETRIEVAL_QUERY`) and the 3 nearest knowledge-base chunks are fetched from Chroma. Only chunks with cosine similarity ≥ 0.68 *and* within 0.05 of the best are kept — off-topic questions retrieve nothing.
3. **Prompt assembly.** The model receives: a system prompt (persona, rules, **today's date** so "tomorrow" resolves), the **conversation history**, and the new question with any retrieved context appended (context is *not* stored in history).
4. **Tool loop.** Gemini either answers in text or emits a *function call* (`cancel_order(order_id="1043")`). The agent executes it through the registry, feeds the result back as a `function_response`, and calls the model again — up to 4 rounds. Every call and result is recorded.
5. **Handoff check.** If the model called `request_human_handoff`, **or** a bilingual keyword check matches ("human", "agent", "ಏಜೆಂಟ್", …), the result carries `handoff=true` plus a summary (reason, remembered slots, tools used, last messages).
6. **Memory.** The turn is appended to a bounded per-session history; tool arguments (order ID, name, date) are stored as slots.
7. **Response** returns `response`, `sources`, `tool_calls` (name, args, result), `confidence`, `handoff`, `handoff_summary`.

### 4.2 A voice turn

`POST /api/voice` (multipart audio) → type/size checks → temp file → **ASR** → transcript → *the same agent path as above* → reply text → **TTS** (language chosen from the reply's script) → mp3 saved → response includes `transcript` and `audio_url` (`/api/audio/{name}`, served over HTTP, auto-expires after 1 h).

### 4.3 Key mechanisms

- **Function calling instead of rules.** The model chooses the tool and arguments, so "cancel it", "ನನ್ನ ಆರ್ಡರ್ ಎಲ್ಲಿದೆ?" or "book me in tomorrow at 10" work without a single regex. Missing arguments (order ID, name) are *asked for*, not guessed.
- **Tool registry = single source of truth.** One `ToolSpec` holds the JSON schema Gemini sees *and* the Python function that runs, so declaration and implementation cannot drift. Tool failures are returned to the model (not raised), letting it explain errors in the user's language.
- **Grounding.** Policy answers must come from retrieved context; the prompt forbids inventing facts and tells the model to offer a human instead.
- **Honest confidence.** `confidence` is the cosine similarity of the best retrieved chunk, or `null` when none was used — never a default.
- **Resilience.** Retries with exponential backoff on 429/5xx; a *daily-quota* 429 fails fast; then an ordered chain of fallback models; retrieval failure degrades to chat-without-RAG; the keyword handoff works even when the LLM is down; the API starts even if initialisation fails and reports the problem on `/api/health`.
- **Data safety.** Seed data in `data/` is copied to a gitignored `.state/` on first use; writes are atomic (temp file + replace) and lock-protected, so using the app never dirties git and never leaves half-written files.

---

## 5. Component notes

| Component | Behaviour worth knowing |
|---|---|
| **Orders** | Only `Processing`/`Placed` orders are cancellable; `Shipped`, `Delivered`, already-`Cancelled` return explanatory errors. IDs normalised (`#1042`, `1042`). |
| **Appointments** | Hourly slots 10:00–16:00. Parses `10am`, `14:00`, `2 PM`; rejects past dates, invalid slots, missing names and double-booking. |
| **Chunking** | Sentence-aware, ≤ 400 chars; the document title is embedded with each chunk (improves routing on a tiny corpus). Ingestion is idempotent (stable IDs + upsert). |
| **Memory** | Last 20 messages per session (always starts on a user turn), 1000 sessions (LRU), thread-safe. |
| **TTS** | Kannada if ≥ 25 % of letters are Kannada script, else English; filenames validated (no path traversal). |
| **UI** | Persistent chat history; tool calls, sources and handoff summary shown under the message; mic recorder + upload; backend status; duplicate-submit guard. |

---

## 6. Testing and results

### 6.1 Automated tests — 81, fully offline
Tools and validation, time/date parsing, chunking, retrieval scoring (floor + margin), memory bounds, handoff detection (English + Kannada, with negatives such as "humanoid"), language detection, prompts, the agent loop with a scripted fake LLM (tool round-trip, history, slot memory, handoff, outage, retrieval failure, runaway-loop cap), LLM retry/fallback/daily-quota behaviour, Gemini ASR, and every API route including voice and path-traversal protection. `ruff` is clean; GitHub Actions CI passes.

### 6.2 Live end-to-end benchmark — 19 cases (English + Kannada)

Final run, single pass on `gemini-3.5-flash-lite` (fallback chain), real embeddings and Chroma:

| Metric | Result |
|---|---|
| Tool accuracy (name **and** arguments) | **100 %** (7/7) |
| No-tool accuracy (greeting, out-of-scope, missing-ID) | **100 %** (3/3) |
| RAG recall@3 (incl. 2 Kannada queries) | **100 %** (6/6) |
| Answer-fact coverage (English) | **100 %** (6/6) |
| Handoff accuracy (incl. a negative case) | **100 %** (3/3) |
| Latency p50 / p95 | **1.9 s / 8.2 s** |

An earlier run on the larger `gemini-3.6-flash` also passed 19/19. Per-case tables and notes: [`docs/evaluation.md`](evaluation.md). The suite is small, so treat it as a regression check, not a general quality benchmark.

### 6.3 Other live checks
- **Voice:** English audio → Whisper → correct tool call → spoken reply ✔. Kannada audio → **Gemini ASR** → exact transcript → tool call → Kannada reply + audio ✔.
- **UI:** Streamlit headless test against the live API: renders, sends a message, shows the answer and its sources, no exceptions ✔.
- **Docker:** image builds; `docker compose up` brings the API to *healthy*, ingests the knowledge base inside the container, and serves the UI ✔.

---

## 7. What was wrong originally, and what live testing uncovered

**Prototype problems fixed:** keyword/regex "tool routing" (the LLM never saw tools) · history stored but never sent to the model · hard-coded `0.95` confidence · fake reply on LLM failure ("I have processed your request.") · imports and Dockerfile disagreed (`vaani.app.*` vs `app.main`), all paths relative to the CWD · voice endpoint omitted fields the UI read and leaked a server file path · duplicate chunks on every ingest · guessed similarity formula · tools mutated tracked JSON files · empty tests and a placeholder "groundedness" metric · `torch`, `sentence-transformers`, `pandas`… installed but unused.

**Found only by running against real Gemini:**
1. The hard-coded model `gemini-2.5-flash` is **retired** — the original app could not have worked.
2. Embedding similarities are compressed (unrelated text ≈ 0.55), so the retrieval floor was tuned empirically to 0.68; order questions had been pulling irrelevant policy chunks.
3. Free-tier quota is **~20 requests/model/day** → daily-quota fast-fail and a fallback chain.
4. **Whisper on CPU cannot do Kannada** (`small` → wrong script; `medium` → inaccurate, ~100 s) → Gemini ASR backend.
5. A lite model refused a cancellation from policy text without calling the tool → prompt rule tightened ("always call the tool; only it knows the real status").
6. Benchmark fact-checks were too literal ("5-7" vs "5 to 7", "2 hours" vs "two hours") → made tolerant.

---

## 8. Known limitations (honest list)

- **No authentication or rate limiting** — anyone who can reach the API can cancel demo orders. Put it behind your own gateway/auth before exposure.
- **State is in-process** — sessions vanish on restart and cannot be shared across instances (use Redis/DB).
- **Orders/appointments are JSON files** — a stand-in; `OrderService`/`AppointmentService` are the seams for a real backend.
- **Tiny knowledge base** (3 documents); retrieval thresholds should be re-tuned when it grows.
- **Gemini free tier** is quota-limited; production use needs billing enabled.
- **Kannada voice quality** was verified on synthesized (gTTS) audio, not on a real human voice; Whisper is unreliable for Kannada.
- **gTTS** needs internet and sounds synthetic; **Streamlit's mic widget** was not exercised in a real browser.
- Data-privacy: audio and text are sent to Google's API.

---

## 9. Where Vaani fits best

Vaani is strongest wherever there is **a bounded set of customer questions and a few safe, well-defined actions**, **customers who switch between a regional language and English**, and **a need for a human escape hatch**.

**Best-fit use cases**
1. **E-commerce / retail support** — order status, cancellation, returns & shipping policy (this is the built-in demo).
2. **Appointment-driven services** — clinics, salons, service centres, banks' branch visits, government offices: check availability and book by voice.
3. **Regional-language contact centres (Karnataka and beyond)** — first-line support in Kannada/English code-switching, with automatic escalation and a ready summary for the human agent. The same design extends to Hindi, Tamil, Telugu by changing the prompt, ASR/TTS language and knowledge base.
4. **Telecom / utilities / insurance FAQs** — plan, bill, claim-status lookups plus policy Q&A grounded in documents.
5. **Voice-first or low-literacy settings** — users who find typing in Kannada script hard can simply speak.
6. **Internal helpdesks** — HR/IT policy Q&A and simple requests (leave balance, ticket status).

**Why it works well there:** answers are grounded in your documents (less hallucination); actions go through validated tools (the model cannot, for example, cancel a shipped order); every decision is inspectable in `tool_calls`/`sources`; and the human-handoff summary shortens resolution time.

**Where it is a poor fit (without more work):** high-stakes advice (medical/legal/financial), payments or identity-verified actions (needs auth, audit logs, consent flows), very large or fast-changing knowledge bases (needs better chunking, hybrid search, reranking), and noisy telephony audio (needs a telephony ASR stack and streaming).

---

## 10. Recommended next steps

1. Authentication, per-user authorisation (a customer may only act on their own orders) and rate limiting.
2. Replace JSON stores with a database/real order API; move sessions to Redis.
3. Streaming responses and streaming ASR/TTS for lower perceived latency; a higher-quality TTS voice.
4. Larger knowledge base with reranking and per-document metadata; periodic re-evaluation.
5. Evaluate Kannada ASR on real recorded speech; add Hindi/Tamil/Telugu.
6. Observability: structured logs, tracing of tool calls, cost/latency dashboards.
7. Human-in-the-loop channel for handoff (ticket/Slack/CRM integration) and an LLM-judge for answer quality.

---

## 11. Running it

```bash
pip install -r requirements.txt
cp .env.example .env                       # set GEMINI_API_KEY
uvicorn vaani.api.app:app                  # API  → http://localhost:8000/docs
streamlit run frontend/app.py              # UI   → http://localhost:8501
# Kannada voice:  VAANI_ASR_BACKEND=gemini
docker compose up --build                  # or run both in containers
pytest && ruff check .                     # 81 tests, lint
python -m vaani.evaluation.benchmark       # live benchmark (use VAANI_STATE_DIR=.state-eval on the server)
```
