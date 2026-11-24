# Evaluation report

Run on 2026-11-24 against `gemini-3.6-flash (fallbacks: gemini-3.5-flash, gemini-3.5-flash-lite)` via `python -m vaani.evaluation.benchmark`.

| Metric | Result |
|---|---|
| Tool accuracy | 100% (7/7) |
| No-tool accuracy | 100% (3/3) |
| RAG recall@3 | 100% (6/6) |
| Fact coverage (EN) | 100% (6/6) |
| Handoff accuracy | 100% (3/3) |
| Cases passed | 100% (19/19) |
| Latency p50 | 3884 ms |
| Latency p95 | 73432 ms |

## Cases

| Case | Type | Lang | Result | Latency |
|---|---|---|---|---|
| return policy | rag | en | PASS | 3471 ms |
| return used item | rag | en | PASS | 13910 ms |
| shipping time | rag | en | PASS | 3193 ms |
| cancellation policy | rag | en | PASS | 73432 ms |
| kn return policy | rag | kn | PASS | 4154 ms |
| kn shipping | rag | kn | PASS | 3052 ms |
| order status | tool | en | PASS | 3884 ms |
| kn order status | tool | kn | PASS | 3301 ms |
| cancel shipped (refused) | tool | en | PASS | 51939 ms |
| check appointments | tool | en | PASS | 7444 ms |
| book appointment | tool | en | PASS | 3419 ms |
| book 'tomorrow' | tool | en | PASS | 13195 ms |
| multi-turn: ask id then act | tool | en | PASS | 12093 ms |
| cancel without id asks first | none | en | PASS | 1915 ms |
| greeting | none | en | PASS | 1715 ms |
| out of scope | none | en | PASS | 12033 ms |
| human request | handoff | en | PASS | 2356 ms |
| kn agent request | handoff | kn | PASS | 1028 ms |
| no handoff on normal query | handoff | en | PASS | 9702 ms |

## How this run was produced

The Gemini free tier allows ~20 generate requests per model per day, so the 19 cases were run in two passes
against the same code: 12 passed in the first pass; the other 7 failed there only because the quota / a 503 spike
was hit (plus one fact check that wrongly required "5-7" instead of accepting "5 to 7") and passed when re-run with
`--only`. Latencies are wall-clock including retries, so the p95 is dominated by quota/overload handling
(first-pass worst cases were 52-73 s while Gemini was returning 503s); typical single-call latency is 2-4 s, and
tool-calling turns (two LLM calls) 3-13 s.

## Voice pipeline (manual check, real audio)

Audio was synthesized with gTTS and posted to `/api/voice`.

| Clip | ASR backend | Transcript | Outcome |
|---|---|---|---|
| English "Where is my order 1043?" | Whisper `small` | exact | `get_order_status("1043")`, spoken reply |
| Kannada "ನನ್ನ ಆರ್ಡರ್ 1043 ರ ಸ್ಥಿತಿ ಏನು?" | Whisper `small` | wrong script (Devanagari), unusable | agent asked for the order ID |
| Kannada (same clip) | Whisper `medium` | close but inaccurate, ~100 s on CPU | not used |
| Kannada (same clip) | **Gemini (`VAANI_ASR_BACKEND=gemini`)** | exact | `get_order_status("1043")`, Kannada reply + audio |

Conclusion: use `VAANI_ASR_BACKEND=gemini` for Kannada; Whisper is the offline option and is reliable for English.
