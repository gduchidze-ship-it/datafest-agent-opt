"""Cells for the condensed 25 minute demo notebook."""

# ruff: noqa: E501
from __future__ import annotations

from scripts.cells import Cell, code, md


def _round(
    header: str,
    diagram: str,
    previous: str,
    current: str,
    page: str | None = None,
    extras: list[Cell] | None = None,
) -> list[Cell]:
    return [
        md(header),
        code(f'draw_diagram("{diagram}")'),
        *(
            [
                code(
                    f'display(request_diff({previous}, {current}, {page}).style.hide(axis="index"))'
                )
            ]
            if page
            else []
        ),
        *(extras or []),
        code(f"""
display(before_after(summary, {previous}, {current}).style.hide(axis="index"))
print(gate(traces, {current}))
display(plot_progress(summary, upto_variant={current}))
"""),
    ]


def _intro() -> list[Cell]:
    return [
        md("""
# Optimizing document review agents for cost and quality: decision models vs LLMs

An e-discovery agent reviews each page: **produce**, **withhold**, **redact** or send to an **attorney**.

**Goal:** cut cost and latency without losing quality, and find which steps need an LLM and which only need a decision model.
"""),
        code("""
%matplotlib inline
import os

import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from openai import AsyncOpenAI

load_dotenv(override=False)
RUN_LIVE_API_CALLS = os.environ.get("RUN_LIVE_API_CALLS", "false").lower() == "true"
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
if RUN_LIVE_API_CALLS and not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before enabling live API calls.")
if LANGSMITH_TRACING and not os.environ.get("LANGSMITH_API_KEY"):
    raise RuntimeError("Set LANGSMITH_API_KEY before enabling LangSmith tracing.")
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGSMITH_TRACING).lower()
os.environ.setdefault("LANGSMITH_PROJECT", "doc-review-opt")
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.live import BudgetTracker
from doc_review_opt.reporting.diagrams import draw_diagram
LIVE = get_settings().live
client = AsyncOpenAI(timeout=LIVE.timeout_s, max_retries=LIVE.max_retries) if RUN_LIVE_API_CALLS else None
budget = BudgetTracker()
print(f"Live: {RUN_LIVE_API_CALLS} | LangSmith: {LANGSMITH_TRACING} | budget stop: ${budget.limit_usd:.2f}")
"""),
        md("""
| Step | Section | Time |
|---|---|---|
| The agent | Business rules, the LangGraph graph, one run | 4 min |
| Version 1 | Success criteria, eval set, baseline | 3 min |
| Traces → find errors | Measure the baseline | 3 min |
| Tune → climb | Rounds 1 to 5, one lever each | 12 min |
| Tradeoffs | LLM vs Decisions | (in Round 3) |
| Check | Results, live reality check, takeaways | 5 min |
"""),
    ]


def _agent() -> list[Cell]:
    return [
        md("""
## The agent

**The business.** In a lawsuit, Arcadia Freight must hand the other side every relevant page. Each page gets exactly one outcome:

| The page is… | Outcome | Why it matters |
|---|---|---|
| Lawyer\u2013client advice (privileged) | **attorney review**, then withhold | Producing it can waive privilege |
| Not relevant to the case | **withhold** | Nothing to hand over |
| Relevant, with personal data | **redact**, then produce | Personal data must not leak |
| Relevant and clean | **produce** | |
| Unclear | **attorney review** | Safe, but $1.50 of attorney time |
"""),
        code("""
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.walkthrough import page_table

OUTCOMES = ("produce", "withhold", "redact", "attorney_review")
demo_pages = [next(p for p in EVAL_SET if p.labels.expected_route == o) for o in OUTCOMES]
display(page_table(demo_pages).style.hide(axis="index"))
"""),
        md("""
## How the agent works: one LangGraph graph

- `Send` fans the batch out: one `review_page` subgraph per page, run in parallel.
- `classify` runs the engine. In Version 1 the agent's own answer picks the end node; from Round 3, plain code (`decide`) picks it from typed answers.
- `attorney_review` pauses the graph with `interrupt()` until an attorney answers. `redact` jumps there only if redaction fails; LangGraph's drawing leaves out its normal exit.

Dashed arrows are conditional. Every round keeps this graph; only what runs inside the nodes changes. Version 1's engine is one `gpt-6.1-terra` agent with five tools.
"""),
        code("""
from doc_review_opt.agent.graph import build_review_graph, review_checkpointer
from doc_review_opt.agent.variants import BASELINE, VARIANTS, default_policy
from doc_review_opt.experiment.simulation import build_simulated_engines
from doc_review_opt.reporting.diagrams import draw_graph

policy = default_policy(VARIANTS[BASELINE])
engine, redactor = build_simulated_engines(VARIANTS[BASELINE], EVAL_SET, policy)
review_graph = build_review_graph(engine, redactor, review_checkpointer(), policy=policy)
draw_graph(review_graph.get_graph(xray=True))
"""),
        md("""
### One run: four pages through Version 1

The privileged page stops at `attorney_review`. The pause carries the page id and the reason, never page text.
"""),
        code("""
from uuid import uuid4

config = {"configurable": {"thread_id": f"intro-{uuid4()}"}}
paused = await review_graph.ainvoke({"pages": demo_pages, "results": []}, config)
waiting = [pending.value["doc_id"] for pending in paused["__interrupt__"]]
print(f"{len(paused['results'])} pages finished, {len(waiting)} waiting for an attorney: {', '.join(waiting)}")
"""),
        code('draw_diagram("human_in_the_loop")'),
        code("""
from langgraph.types import Command

resume = {pending.id: "withhold" for pending in paused["__interrupt__"]}
final = await review_graph.ainvoke(Command(resume=resume), config)
display(page_table(demo_pages, final["results"]).style.hide(axis="index"))
"""),
    ]


def _loop() -> list[Cell]:
    return [
        md("""
## The optimization loop

Version 1 handles these four pages. Is it cheap, fast and safe on all 30? Measure, find errors, change one lever, measure again.
"""),
        code('draw_diagram("optimization_loop")'),
    ]


def _version_one() -> list[Cell]:
    return [
        md("""
## Success criteria

- **Hard gates:** zero privileged pages produced, zero PII left in produced text.
- **Accept a round** only if at least 95% of pages pass the guardrails.
- **Cost** = model cost + attorney time ($1.50 per escalated page).
"""),
        md("""
## Eval set

30 synthetic, labelled pages for one matter; 10 are traps or borderline.
"""),
        code("""
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.reporting import plot_eval_set

eval_table = pd.DataFrame(
    {
        "doc_id": page.doc_id,
        "expected_route": page.labels.expected_route,
        "difficulty": page.labels.difficulty,
    }
    for page in EVAL_SET
)
display(plot_eval_set(eval_table))
"""),
        md("""
## Version 1: the bad baseline

One `gpt-6.1-terra` super-agent: every tool on every page, verbose payloads, free-text answer.
"""),
        code('draw_diagram("bad_baseline")'),
    ]


def _measure() -> list[Cell]:
    return [
        md("""
## Measure: traces, then errors

Every round runs on all 30 pages (simulated, seconds). One trace per page per round.
"""),
        code('draw_diagram("eval_pipeline")'),
        md("""
- **Run and trace:** each variant goes through the same graph; every page leaves one trace.
- **Grade:** deterministic guardrails and metrics read the trace and the page labels. No LLM judge.
- **Gate and report:** hard gates and the guardrail pass floor accept or reject the round; reported metrics only warn, marked **(watch)** below a floor in `config.toml`.
"""),
        code("""
from doc_review_opt.agent.variants import (
    BASELINE,
    CACHING,
    CALIBRATED_POLICY,
    CONTROLS,
    DECISION_MODEL,
    ROUTING,
    SPLIT,
    VARIANT_ORDER,
)
"""),
        code("""
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import traces_frame
from doc_review_opt.reporting.reporting import plot_progress, summarize
from doc_review_opt.reporting.walkthrough import before_after, gate, metrics_table, request_diff, symptoms

traces = await simulate_all()
frame = traces_frame(traces)
summary = summarize(frame)
SAMPLE_PAGE = next(p for p in EVAL_SET if p.labels.privileged)
LOW_PAGE = next(p for p in EVAL_SET if p.labels.risk == "low" and not p.labels.privileged)
"""),
        code("""
columns = ["variant_label", "doc_id", "route", "tool_calls", "sync_tokens", "latency_s", "model_cost_usd", "quality_score"]
formats = {"sync_tokens": "{:,.0f}", "latency_s": "{:.2f}", "model_cost_usd": "${:.5f}", "quality_score": "{:.2f}"}
display(frame[columns].head().style.format(formats).hide(axis="index"))
"""),
        code("""
print(symptoms(summary, BASELINE))
print(gate(traces, BASELINE))
"""),
        code("""
display(metrics_table(traces, [BASELINE]).style.hide(axis="index"))
"""),
        md("""
**Errors found:** 4.2 of 5 tool calls per page are unneeded, about 8k tokens per page, PII left in produced text on 2 pages, route accuracy 77%, 25% needless attorney reviews. Each round fixes one thing.
"""),
    ]


def _round_one() -> list[Cell]:
    return _round(
        """
## Round 1: prompt, tool and context controls

**Lever:** tight prompt, allow-listed tools, slim payloads, strict JSON answer.
""",
        "round1",
        "BASELINE",
        "CONTROLS",
        "SAMPLE_PAGE",
    )


def _round_two() -> list[Cell]:
    return _round(
        """
## Round 2: model routing

**Lever:** `gpt-6-luna` triages; only risky pages go to `gpt-6.1-terra`.

**Watch:** the cheapest round so far, but it misses one required escalation.
""",
        "round2",
        "CONTROLS",
        "ROUTING",
        "LOW_PAGE",
    )


def _round_three_extras() -> list[Cell]:
    return [
        md("### Three typed questions, one call"),
        code("""
from doc_review_opt.agent.classifiers.decisions import QUESTIONS, DecisionsClassifier
from doc_review_opt.agent.routing import decide

questions = pd.DataFrame(
    {"name": q["name"], "type": q["type"], "instructions": q["instructions"][:90] + ("…" if len(q["instructions"]) > 90 else "")}
    for q in QUESTIONS
)
display(questions.style.hide(axis="index"))
"""),
        code("""
from doc_review_opt.experiment.live import BudgetedClassifier
from doc_review_opt.experiment.simulation import SimulatedTypedClassifier
from doc_review_opt.fixtures import PROTOCOL

if RUN_LIVE_API_CALLS:
    classifier = BudgetedClassifier(DecisionsClassifier(client), budget)
else:
    classifier = SimulatedTypedClassifier("decisions")
result = await classifier.classify(SAMPLE_PAGE, PROTOCOL)
"""),
        code("""
mode = "live" if RUN_LIVE_API_CALLS else "simulated"
print(f"{SAMPLE_PAGE.doc_id} ({mode}): {result.latency_s:.2f}s, ${result.cost_usd:.5f}")
if result.error:
    print(result.error)
else:
    shown = (("responsive score (0 to 2)", "responsive", "score"), ("privileged probability", "privileged", "probability"), ("contains-PII probability", "contains_pii", "probability"))
    for label, name, field in shown:
        value = getattr(result.answers.get(name), field, None)
        print(f"{label}: {'refused' if value is None else f'{value:.2f}'}")
decision = decide(result, CALIBRATED_POLICY)
print(f"route -> {decision.route} ({decision.reason})")
"""),
        md("### The router is plain code: readable, testable, auditable"),
        code("import inspect\n\nprint(inspect.getsource(decide))"),
        md(
            "A refusal or an error never produces a page. It fails safe to an attorney."
        ),
        code("""
from doc_review_opt.agent.classifiers.decisions import DECISIONS_MODEL
from doc_review_opt.models import ClassifierResult, RefusalAnswer

refused = ClassifierResult(
    engine="decisions",
    model=DECISIONS_MODEL,
    answers={"privileged": RefusalAnswer(name="privileged")},
)
print(decide(refused))
"""),
    ]


def _round_three() -> list[Cell]:
    return _round(
        """
## Round 3: replace the LLM with a decision model

**Lever:** one Decisions call answers three typed questions; plain code routes the page; `gpt-6.1-terra` gives a second opinion on the ambiguous band only.

**Watch:** model cost per page rises from Round 2, but the missed escalation is fixed and p50 latency halves.
""",
        "round3",
        "ROUTING",
        "DECISION_MODEL",
        "SAMPLE_PAGE",
        _round_three_extras(),
    )


def _tradeoffs() -> list[Cell]:
    return [
        md("""
## LLM vs Decisions: the tradeoffs

Same three questions, four engines, 30 pages (simulated, calibrated thresholds).
"""),
        code("""
def show_engines(table):
    view = table[["label", "accuracy_vs_labels", "cost_per_page_usd", "p50_latency_s", "escalation_rate", "privileged_leaks"]]
    view = view.rename(columns={"label": "Engine", "accuracy_vs_labels": "Accuracy", "cost_per_page_usd": "$ / page", "p50_latency_s": "p50 (s)", "escalation_rate": "To attorney", "privileged_leaks": "Privileged leaks"})
    formats = {"Accuracy": "{:.0%}", "$ / page": "${:.5f}", "p50 (s)": "{:.2f}", "To attorney": "{:.0%}"}
    display(view.style.format(formats).hide(axis="index"))
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import engine_comparison_table, plot_engine_tradeoff, simulate_engine_comparison

engine_runs = await simulate_engine_comparison(policy=CALIBRATED_POLICY)
engines = engine_comparison_table(engine_runs)
show_engines(engines)
display(plot_engine_tradeoff(engines))
"""),
        md("""
**Read it:** Decisions alone is the fastest and cheapest, but it sends more pages to an attorney (50% vs 33%). That is why Round 3 adds the `gpt-6.1-terra` second opinion on the ambiguous band only.
"""),
        md("""
### Probabilities are a dial

Move the privilege threshold. Decisions never leaks a privileged page; the LLMs' self-reported probabilities start leaking.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import plot_threshold_sweep, threshold_sweep

display(plot_threshold_sweep(threshold_sweep(await simulate_engine_comparison(), EVAL_SET)))
"""),
        md("""
### What Decisions can't do

Write text, extract spans or call tools. Each step gets the engine it needs:
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import STEP_MATRIX

display(pd.DataFrame(STEP_MATRIX).style.hide(axis="index"))
"""),
        code('draw_diagram("decisions_or_llm")'),
    ]


def _round_four_five() -> list[Cell]:
    return [
        *_round(
            """
## Round 4: prompt caching

**Lever:** the stable protocol prefix goes first, with one cache key per matter.

**Watch:** cache writes cost 1.25x, so caching pays only at volume.
""",
            "round4",
            "DECISION_MODEL",
            "CACHING",
            "SAMPLE_PAGE",
        ),
        *_round(
            """
## Round 5: split the workflow

**Lever:** privilege log, QA and tags leave the page path and go to Batch at 50% off.
""",
            "round5",
            "CACHING",
            "SPLIT",
            None,
        ),
    ]


def _wrapup() -> list[Cell]:
    return [
        md("## Results: the climb"),
        code("""
from doc_review_opt.reporting.reporting import plot_quality_gate, plot_round_chart
from doc_review_opt.reporting.walkthrough import confusion_table, gate_table, holdout_table, metrics_table

floor = get_settings().quality.guardrail_pass_floor
display(plot_round_chart(summary))
display(plot_quality_gate(gate_table(traces, VARIANT_ORDER), floor))
"""),
        code("""
display(metrics_table(traces, VARIANT_ORDER).style.hide(axis="index"))
display(confusion_table(traces, SPLIT))
"""),
        code("""
from doc_review_opt.experiment.metrics import classifications_by_page, holdout_check

report = holdout_check(classifications_by_page(engine_runs["decisions"]))
display(holdout_table(report).style.hide(axis="index"))
print(f"Leak-free on the unseen half: {report.holds} ({report.calibration_pages} and {report.holdout_pages} pages)")
"""),
        md("""
**Calibration holdout:** the policy is refit on one half of the pages (alternating within each route) and scored on the other. No leaks on unseen pages, and far cheaper than the default policy (default: 81% and 86% escalations). Escalations are 44% on the calibration half and 57% on the holdout half, and quality is 0.93 and 0.89. The thresholds were tuned on the calibration half, so only the holdout rows are out of sample; the holdout is only 14 pages, a small sample.
"""),
        md("""
## Live reality check

Real APIs, 6 pages, 2026-11-18, $0.18 total.

| Engine | Accuracy | $ / page | p50 |
|---|---|---|---|
| **Decisions** | 100% | $0.00019 | **0.38s** |
| `gpt-6-luna` LLM | 100% | **$0.00011** | 2.26s |
| `gpt-6.1-terra` LLM | 100% | $0.00073 | 3.03s |
| `gpt-6.1-terra` agent | 100% | $0.00118 | 2.33s |

Decisions is **about 6x faster** than `gpt-6-luna` and **4 to 6x cheaper** than `gpt-6.1-terra`, but **1.7x pricier** than `gpt-6-luna`.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_sim_vs_live, recorded_live_note, recorded_live_summary

display(plot_sim_vs_live(summary, recorded_live_summary(), recorded_live_note()))
"""),
        md("""
Optional, on stage: rerun the engine comparison on the real APIs. It takes about a minute, costs a few cents and sends LangSmith traces.
"""),
        code("""
from itertools import zip_longest

from doc_review_opt.experiment.live import LIVE_PAGE_LIMIT
from doc_review_opt.experiment.tradeoffs import live_engine_comparison

by_route = {}
for page in EVAL_SET:
    by_route.setdefault(page.labels.expected_route, []).append(page)
live_pages = [p for group in zip_longest(*by_route.values()) for p in group if p]
live_pages = live_pages[:LIVE_PAGE_LIMIT]
"""),
        code("""
live_runs = {}
if RUN_LIVE_API_CALLS:
    live_runs = await live_engine_comparison(live_pages, client=client, budget=budget)
if live_runs:
    show_engines(engine_comparison_table(live_runs))
    print("\\n".join(f"{n}: {r.trace_url or 'LangSmith tracing is off'}" for n, r in live_runs.items()))
    print(f"Spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
elif RUN_LIVE_API_CALLS:
    print(f"Stopped by the budget: ${budget.total_usd:.2f} of ${budget.limit_usd:.2f}")
else:
    print("Offline. Set RUN_LIVE_API_CALLS=true to rerun this on the real APIs.")
"""),
        md("""
## Takeaways

1. **Measure first.** A labelled eval set and hard gates before any tuning.
2. **Cut waste before changing models.** Controls gave the biggest drop.
3. **Cheapest is not best.** Every round must pass the quality gate.
4. **Typed decision? Try Decisions with a code router.** Fast, calibratable, auditable.
5. **Need text, tools or extraction? Keep an LLM**, the smallest that holds quality.
6. **Re-measure live.** Live, Decisions was fastest, but the smallest LLM was cheaper.
"""),
        md("""
## References

Last verified: November 18, 2026.

- [Optimizing agents for cost and quality](https://github.com/openai/openai-cookbook/tree/main/examples/agent_optimization) (OpenAI Cookbook)
- [Document review agent](https://gist.github.com/sydney-runkle/a632ba4ea0b2b72501dfa4b6ab2a7d8a) (gist by Sydney Runkle)
- [Decisions API guide](https://developers.openai.com/api/docs/guides/decisions)
- [Pricing](https://developers.openai.com/api/docs/pricing)
- LangGraph [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) and [`interrupt`](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
"""),
    ]


def cells() -> list[Cell]:
    return [
        *_intro(),
        *_agent(),
        *_loop(),
        *_version_one(),
        *_measure(),
        *_round_one(),
        *_round_two(),
        *_round_three(),
        *_tradeoffs(),
        *_round_four_five(),
        *_wrapup(),
    ]
