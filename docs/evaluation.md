# Evaluation report

Run on 2026-11-25 against `gemini-3.5-flash-lite (fallbacks: gemini-flash-lite-latest, gemini-3.1-flash-lite)` via `python -m vaani.evaluation.benchmark`.

| Metric | Result |
|---|---|
| Tool accuracy | 100% (7/7) |
| No-tool accuracy | 100% (3/3) |
| RAG recall@3 | 100% (6/6) |
| Fact coverage (EN) | 100% (6/6) |
| Handoff accuracy | 100% (3/3) |
| Cases passed | 100% (19/19) |
| Latency p50 | 1886 ms |
| Latency p95 | 8226 ms |

## Cases

| Case | Type | Lang | Result | Latency |
|---|---|---|---|---|
| return policy | rag | en | PASS | 3530 ms |
| return used item | rag | en | PASS | 1336 ms |
| shipping time | rag | en | PASS | 1172 ms |
| cancellation policy | rag | en | PASS | 1056 ms |
| kn return policy | rag | kn | PASS | 1372 ms |
| kn shipping | rag | kn | PASS | 1125 ms |
| order status | tool | en | PASS | 1985 ms |
| kn order status | tool | kn | PASS | 1794 ms |
| cancel shipped (refused) | tool | en | PASS | 2471 ms |
| check appointments | tool | en | PASS | 2223 ms |
| book appointment | tool | en | PASS | 1661 ms |
| book 'tomorrow' | tool | en | PASS | 1777 ms |
| multi-turn: ask id then act | tool | en | PASS | 2851 ms |
| cancel without id asks first | none | en | PASS | 2337 ms |
| greeting | none | en | PASS | 5704 ms |
| out of scope | none | en | PASS | 6019 ms |
| human request | handoff | en | PASS | 8226 ms |
| kn agent request | handoff | kn | PASS | 1886 ms |
| no handoff on normal query | handoff | en | PASS | 1376 ms |

## How this run was produced

All 19 cases ran in a single pass against the final code on the **lite** model chain, because the larger models'
free-tier daily quota (~20 requests/model/day) was exhausted. One case ("cancellation policy") was flagged by the
benchmark only because its fact check required the literal "2 hours" while the correct reply said "two hours"; the
checker now accepts spelled-out numbers and that case was re-run on its own (passes, 1.1 s). An earlier run on
`gemini-3.6-flash` also passed 19/19, but in two passes for the same quota reason, and surfaced one real prompt gap on
the lite model (it refused a cancellation from policy text without calling the tool); that rule was tightened and
verified. Latencies are wall-clock for the whole conversation turn including tool round-trips (a tool-calling turn is two LLM calls).

## Voice pipeline (manual check, real audio)

Audio was synthesized with gTTS and posted to `/api/voice`.

| Clip | ASR backend | Transcript | Outcome |
|---|---|---|---|
| English "Where is my order 1043?" | Whisper `small` | exact | `get_order_status("1043")`, spoken reply |
| Kannada "ನನ್ನ ಆರ್ಡರ್ 1043 ರ ಸ್ಥಿತಿ ಏನು?" | Whisper `small` | wrong script (Devanagari), unusable | agent asked for the order ID |
| Kannada (same clip) | Whisper `medium` | close but inaccurate, ~100 s on CPU | not used |
| Kannada (same clip) | **Gemini (`VAANI_ASR_BACKEND=gemini`)** | exact | `get_order_status("1043")`, Kannada reply + audio |

Conclusion: use `VAANI_ASR_BACKEND=gemini` for Kannada; Whisper is the offline option and is reliable for English.
