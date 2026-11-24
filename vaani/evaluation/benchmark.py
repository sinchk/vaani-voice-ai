"""End-to-end benchmark against a running Vaani API.

    uvicorn vaani.api.app:app &
    python -m vaani.evaluation.benchmark --report docs/evaluation.md

Cases that mutate data (booking) write to the server's state directory, so run
the server with ``VAANI_STATE_DIR=.state-eval`` to keep your demo data clean.

Metrics
  tool accuracy      the expected tool was called with the expected arguments
  no-tool accuracy   questions that need no tool did not trigger one
  RAG recall@k       the expected policy file is among the retrieved sources
  fact coverage      the reply states the expected fact (checked on English replies)
  handoff accuracy   handoff is raised exactly when it should be
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx

TOMORROW = (date.today() + timedelta(days=1)).isoformat()
NEXT_WEEK = (date.today() + timedelta(days=7)).isoformat()


@dataclass
class Case:
    name: str
    turns: list[str]  # all turns share one session; checks apply to the final turn
    kind: str  # tool | rag | none | handoff
    lang: str = "en"
    tool: str | None = None
    args: dict[str, str] = field(default_factory=dict)  # expected subset of the tool's arguments
    source: str | None = None
    facts: list[str] = field(default_factory=list)  # any-of; substrings expected in an English reply
    handoff: bool = False


CASES: list[Case] = [
    # --- RAG -----------------------------------------------------------------
    Case("return policy", ["What is your return policy?"], "rag", source="return_policy.txt", facts=["30 days"]),
    Case(
        "return used item",
        ["Can I return a used item?"],
        "rag",
        source="return_policy.txt",
        facts=["unused", "original packaging"],
    ),
    Case("shipping time", ["How long does shipping take?"], "rag", source="shipping_policy.txt", facts=["5-7", "2-3"]),
    Case(
        "cancellation policy",
        ["Help me with the cancellation policy"],
        "rag",
        source="cancellation_policy.txt",
        facts=["2 hours"],
    ),
    Case("kn return policy", ["ವಾಪಸಾತಿ ನೀತಿ ತಿಳಿಸಿ"], "rag", lang="kn", source="return_policy.txt"),
    Case("kn shipping", ["ಶಿಪ್ಪಿಂಗ್ ಎಷ್ಟು ದಿನ ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ?"], "rag", lang="kn", source="shipping_policy.txt"),
    # --- Tools ---------------------------------------------------------------
    Case(
        "order status",
        ["Where is my order 1042?"],
        "tool",
        tool="get_order_status",
        args={"order_id": "1042"},
        facts=["shipped"],
    ),
    Case(
        "kn order status",
        ["ನನ್ನ ಆರ್ಡರ್ 1042 ರ ಸ್ಥಿತಿ ಏನು?"],
        "tool",
        lang="kn",
        tool="get_order_status",
        args={"order_id": "1042"},
    ),
    Case(
        "cancel shipped (refused)",
        ["Please cancel order 1042"],
        "tool",
        tool="cancel_order",
        args={"order_id": "1042"},
        facts=["shipped", "cannot", "can't", "unable"],
    ),
    Case(
        "check appointments",
        [f"Is there an appointment slot free on {NEXT_WEEK}?"],
        "tool",
        tool="check_appointment",
        args={"date": NEXT_WEEK},
    ),
    Case(
        "book appointment",
        [f"Book an appointment for Sinchana on {TOMORROW} at 10 AM"],
        "tool",
        tool="book_appointment",
        args={"date": TOMORROW, "customer": "Sinchana"},
    ),
    Case(
        "book 'tomorrow'",
        ["Book an appointment for Meera tomorrow at 3 PM"],
        "tool",
        tool="book_appointment",
        args={"date": TOMORROW, "customer": "Meera"},
    ),
    Case(
        "multi-turn: ask id then act",
        ["I want to check my order", "It's 1043"],
        "tool",
        tool="get_order_status",
        args={"order_id": "1043"},
    ),
    # --- Should not call tools -----------------------------------------------
    Case("cancel without id asks first", ["I want to cancel my order"], "none"),
    Case("greeting", ["Hi there!"], "none"),
    Case("out of scope", ["What is the capital of France?"], "none"),
    # --- Handoff ---------------------------------------------------------------
    Case("human request", ["I want to talk to a human agent"], "handoff", handoff=True),
    Case("kn agent request", ["ನನಗೆ ಒಬ್ಬ ಏಜೆಂಟ್ ಬೇಕು"], "handoff", lang="kn", handoff=True),
    Case("no handoff on normal query", ["What is your return policy?"], "handoff", handoff=False),
]


def run_case(client: httpx.Client, case: Case) -> dict:
    session = f"eval-{uuid.uuid4().hex[:8]}"
    resp, latencies = {}, []
    for turn in case.turns:
        t0 = time.perf_counter()
        r = client.post("/api/chat", json={"session_id": session, "message": turn})
        latencies.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        resp = r.json()

    calls = resp["tool_calls"]
    reply = resp["response"].lower()
    out = {
        "name": case.name,
        "kind": case.kind,
        "lang": case.lang,
        "latency_ms": sum(latencies),
        "reply": resp["response"],
        "pass": {},
    }

    if case.kind == "tool":
        matching = [c for c in calls if c["name"] == case.tool]
        ok_args = any(all(str(c["args"].get(k)).lower() == v.lower() for k, v in case.args.items()) for c in matching)
        out["pass"]["tool"] = ok_args
    if case.kind == "none":
        out["pass"]["no_tool"] = not [c for c in calls if c["name"] != "request_human_handoff"]
    if case.kind == "rag":
        out["pass"]["rag"] = case.source in [s["source"] for s in resp["sources"]]
    if case.facts and case.lang == "en":
        out["pass"]["fact"] = any(f.lower() in reply for f in case.facts)
    if case.kind == "handoff":
        out["pass"]["handoff"] = resp["handoff"] == case.handoff
    out["ok"] = all(out["pass"].values())
    return out


def pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})" if values else "n/a"


def summarize(results: list[dict]) -> dict:
    def metric(key):
        return [r["pass"][key] for r in results if key in r["pass"]]

    lat = sorted(r["latency_ms"] for r in results)
    return {
        "Tool accuracy": pct(metric("tool")),
        "No-tool accuracy": pct(metric("no_tool")),
        "RAG recall@3": pct(metric("rag")),
        "Fact coverage (EN)": pct(metric("fact")),
        "Handoff accuracy": pct(metric("handoff")),
        "Cases passed": pct([r["ok"] for r in results]),
        "Latency p50": f"{statistics.median(lat):.0f} ms",
        "Latency p95": f"{lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f} ms",
    }


def write_report(path: Path, results: list[dict], summary: dict, model: str) -> None:
    lines = [
        "# Evaluation report",
        "",
        f"Run on {date.today().isoformat()} against `{model}` via `python -m vaani.evaluation.benchmark`.",
        "",
        "| Metric | Result |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "## Cases",
        "",
        "| Case | Type | Lang | Result | Latency |",
        "|---|---|---|---|---|",
        *[
            f"| {r['name']} | {r['kind']} | {r['lang']} | {'PASS' if r['ok'] else 'FAIL'} | {r['latency_ms']:.0f} ms |"
            for r in results
        ],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a running Vaani API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between cases (rate-limit friendly)")
    parser.add_argument("--report", type=Path, help="write a markdown report to this path")
    parser.add_argument("--json", type=Path, help="write raw results to this path")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        try:
            health = client.get("/api/health").json()
        except httpx.HTTPError as exc:
            print(f"Cannot reach {args.base_url}: {exc}\nStart it with: uvicorn vaani.api.app:app", file=sys.stderr)
            return 2
        results = []
        for case in CASES:
            try:
                res = run_case(client, case)
            except httpx.HTTPError as exc:
                res = {
                    "name": case.name,
                    "kind": case.kind,
                    "lang": case.lang,
                    "latency_ms": 0,
                    "reply": f"ERROR: {exc}",
                    "pass": {"request": False},
                    "ok": False,
                }
            results.append(res)
            print(f"{'PASS' if res['ok'] else 'FAIL'}  {res['name']:<32} {res['latency_ms']:>6.0f} ms")
            if not res["ok"]:
                print(f"      checks={res['pass']}  reply={res['reply'][:120]!r}")
            time.sleep(args.delay)

    summary = summarize(results)
    print("\n" + "\n".join(f"{k:<20} {v}" for k, v in summary.items()))
    if args.report:
        write_report(args.report, results, summary, health.get("model", "?"))
    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
