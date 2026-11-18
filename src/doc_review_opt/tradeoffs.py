"""Decisions versus LLM: step matrix, engine comparison, sweep, cascade, calibration."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from openai import AsyncOpenAI

from doc_review_opt.config import Calibration, get_settings
from doc_review_opt.decisions import (
    DEFAULT_POLICY,
    DecisionsClassifier,
    RoutingPolicy,
    decide,
    flags_pii,
)
from doc_review_opt.exceptions import BudgetExceededError
from doc_review_opt.graph import Classifier, OpenAIRedactor
from doc_review_opt.live_api import (
    LIVE_PAGE_LIMIT,
    BudgetTracker,
    LiveAgentClassifier,
    run_traced,
)
from doc_review_opt.llm_classifier import LLMProbabilityClassifier
from doc_review_opt.models import ClassifierResult, ReviewPage, TokenUsage
from doc_review_opt.pricing import approx_tokens, estimate_cost_usd
from doc_review_opt.review_agent import EVAL_SET, REDACTION_PROMPT, scrub_pii
from doc_review_opt.runs import ReviewRun, review_pages, scripted_attorney_decisions
from doc_review_opt.simulation import (
    SimulatedAgentClassifier,
    SimulatedRedactor,
    SimulatedTypedClassifier,
    simulate_variant,
)
from doc_review_opt.traces import (
    ATTORNEY_REVIEW_COST_USD_PER_PAGE,
    PageOutcome,
    Trace,
    evaluate_trace_quality,
    traces_frame,
)
from doc_review_opt.variants import (
    CALIBRATED_POLICY,
    CONTROLS,
    DECISION_MODEL,
    ROUTER_ONLY,
    VARIANTS,
)

logger = logging.getLogger(__name__)

SETTINGS = get_settings()
LIVE = SETTINGS.live
MODELS = SETTINGS.models
MONTHLY_PAGES = SETTINGS.scoring.monthly_pages
ENGINE_LABELS = {
    "decisions": f"Decisions ({MODELS.decisions})",
    "llm-luna": f"LLM probabilities ({MODELS.luna})",
    "llm-terra": f"LLM probabilities ({MODELS.terra})",
    "agent": f"Agent ({MODELS.terra})",
}
TYPED_ENGINES = ("decisions", "llm-luna", "llm-terra")

STEP_MATRIX = [
    {
        "step": "Responsiveness rubric",
        "fits": "Decisions",
        "why": "A fixed 0-2 rubric is a typed score: one cheap call, no text to write.",
    },
    {
        "step": "Privilege predicate",
        "fits": "Decisions",
        "why": "A yes/no with a probability feeds a low escalation threshold; the "
        "cascade sends the ambiguous band to a stronger model.",
    },
    {
        "step": "PII presence",
        "fits": "Decisions",
        "why": "Deciding whether personal data is present is a predicate; finding "
        "the spans is not (Decisions cannot extract to your own schema).",
    },
    {
        "step": "Near-duplicate check",
        "fits": "Code",
        "why": "An index lookup is deterministic and free; no model should do it.",
    },
    {
        "step": "Escalation gate",
        "fits": "Code",
        "why": "Thresholds over typed answers are plain code you can test and audit.",
    },
    {
        "step": "Redaction rewrite",
        "fits": "LLM",
        "why": "Writing the page with spans replaced is generation; Decisions only "
        "answers questions.",
    },
    {
        "step": "Privilege-log drafting",
        "fits": "LLM",
        "why": "A log entry is free text with an explanation; Decisions cannot "
        "write one.",
    },
    {
        "step": "QA tags",
        "fits": "LLM",
        "why": "Tags and notes are generated text; batch them off the page path.",
    },
]
WHEN_TO_USE = [
    {
        "factor": "Output",
        "use_decisions_when": "The answer is a typed route: yes/no, choice or score.",
        "use_an_llm_when": "The output is generated text or your own extraction "
        "schema.",
    },
    {
        "factor": "Volume",
        "use_decisions_when": "Millions of pages and the alternative is a large LLM "
        "(live, 6 pages: 4-6x cheaper than gpt-6.1-terra; about 1.7x pricier than "
        "gpt-6-luna as an LLM).",
        "use_an_llm_when": "Volume is low enough that per-call cost does not matter, "
        "or the smallest LLM is accurate enough.",
    },
    {
        "factor": "Latency",
        "use_decisions_when": "The page path is on a deadline (live, 6 pages: ~6x "
        "faster than gpt-6-luna and 6-8x faster than gpt-6.1-terra at p50).",
        "use_an_llm_when": "A second or two per page is acceptable.",
    },
    {
        "factor": "Tools",
        "use_decisions_when": "No tool calls are needed; code does the lookups.",
        "use_an_llm_when": "The model has to choose and call tools.",
    },
    {
        "factor": "Explanations",
        "use_decisions_when": "A probability is enough for routing.",
        "use_an_llm_when": "A reviewer needs a reason, such as a privilege-log entry.",
    },
    {
        "factor": "Model choice",
        "use_decisions_when": "gpt-6-luna is accurate enough for the rubric.",
        "use_an_llm_when": "You need a stronger model; Decisions only supports "
        "gpt-6-luna.",
    },
    {
        "factor": "Refusals",
        "use_decisions_when": "You treat a refused answer as escalate-to-attorney.",
        "use_an_llm_when": "You can retry or rephrase on a refusal.",
    },
    {
        "factor": "Confidence",
        "use_decisions_when": "You recalibrate thresholds on your own labelled split.",
        "use_an_llm_when": "You cannot recalibrate: confidence does not port between "
        "engines.",
    },
]


@dataclass(frozen=True)
class Replayed:
    outcome: PageOutcome
    trace_cost_usd: float
    privileged_leak: bool
    pii_leak: bool
    quality: float


@dataclass(frozen=True)
class PolicyScore:
    escalation_rate: float
    privileged_leaks: int
    pii_leaks: int
    mean_quality: float
    blended_cost_per_page_usd: float


class NoFeasiblePolicyError(ValueError):
    pass


def _redaction_cost(page: ReviewPage) -> float:
    usage = TokenUsage(
        input=approx_tokens(REDACTION_PROMPT) + approx_tokens(page.text),
        output=approx_tokens(scrub_pii(page)),
    )
    return estimate_cost_usd(MODELS.redaction, usage)


def replay(
    page: ReviewPage, result: ClassifierResult, policy: RoutingPolicy
) -> Replayed:
    route = decide(result, policy).route
    escalated = route == "attorney_review"
    produced = route in ("produce", "redact")
    if escalated:
        produced = scripted_attorney_decisions([page])[page.doc_id] == "produce"
    redacted = produced and (
        route == "redact" or (escalated and flags_pii(result, policy))
    )
    text = None
    if produced:
        text = scrub_pii(page) if redacted else page.text
    outcome = PageOutcome(
        route=route,
        decision="produce" if produced else "withhold",
        escalated=escalated,
        produced_text=text,
        tools=result.tools,
        parse_failed=result.parse_failed,
    )
    quality = evaluate_trace_quality(page, outcome)
    cost = result.cost_usd + (_redaction_cost(page) if redacted else 0.0)
    cost += ATTORNEY_REVIEW_COST_USD_PER_PAGE if escalated else 0.0
    return Replayed(
        outcome, cost, quality.privileged_leak, quality.pii_leak, quality.quality_score
    )


def evaluate_policy(
    results: Mapping[str, ClassifierResult],
    pages: Sequence[ReviewPage],
    policy: RoutingPolicy,
) -> PolicyScore:
    replayed = [replay(p, results[p.doc_id], policy) for p in pages]
    count = len(replayed)
    return PolicyScore(
        escalation_rate=sum(r.outcome.escalated for r in replayed) / count,
        privileged_leaks=sum(r.privileged_leak for r in replayed),
        pii_leaks=sum(r.pii_leak for r in replayed),
        mean_quality=sum(r.quality for r in replayed) / count,
        blended_cost_per_page_usd=sum(r.trace_cost_usd for r in replayed) / count,
    )


def _distance_from_default(policy: RoutingPolicy) -> float:
    return sum(
        abs(float(getattr(policy, name)) - float(getattr(DEFAULT_POLICY, name)))
        for name in DEFAULT_POLICY.__dataclass_fields__
    )


def calibrate_policy(
    results: Mapping[str, ClassifierResult],
    pages: Sequence[ReviewPage],
    grid: Calibration = SETTINGS.calibration,
) -> RoutingPolicy:
    candidates = itertools.product(
        grid.privilege_escalation,
        grid.pii_redaction,
        grid.not_responsive,
        grid.responsive,
        grid.responsive_confidence,
    )
    scored: list[tuple[float, float, RoutingPolicy]] = []
    for values in candidates:
        policy = RoutingPolicy(*values)
        score = evaluate_policy(results, pages, policy)
        if score.privileged_leaks == 0 and score.pii_leaks == 0:
            scored.append(
                (
                    round(score.blended_cost_per_page_usd, 9),
                    _distance_from_default(policy),
                    policy,
                )
            )
    if not scored:
        raise NoFeasiblePolicyError(
            "no policy in the grid has zero privileged and PII leaks"
        )
    return min(scored, key=lambda item: item[:2])[2]


def policy_report(
    results: Mapping[str, ClassifierResult],
    splits: Mapping[str, Sequence[ReviewPage]],
    policies: Mapping[str, RoutingPolicy],
) -> pd.DataFrame:
    rows = []
    for (split, pages), (name, policy) in itertools.product(
        splits.items(), policies.items()
    ):
        score = evaluate_policy(results, pages, policy)
        rows.append({"policy": name, "split": split, **score.__dict__})
    return pd.DataFrame(rows)


def _comparison_spec(name: str) -> Any:
    return replace(VARIANTS[CONTROLS], variant=name, label=ENGINE_LABELS[name])


def _simulated_engine(name: str) -> Classifier:
    if name == "agent":
        return SimulatedAgentClassifier(VARIANTS[CONTROLS])
    return SimulatedTypedClassifier(name)


async def simulate_engine_comparison(
    pages: Sequence[ReviewPage] = EVAL_SET, *, policy: RoutingPolicy = DEFAULT_POLICY
) -> dict[str, ReviewRun]:
    runs = {}
    for name in ENGINE_LABELS:
        runs[name] = await review_pages(
            _comparison_spec(name),
            pages,
            _simulated_engine(name),
            SimulatedRedactor(),
            policy=policy,
            mode="simulated",
        )
    return runs


async def live_engine_comparison(
    pages: Sequence[ReviewPage],
    *,
    client: AsyncOpenAI,
    budget: BudgetTracker,
) -> dict[str, ReviewRun]:
    client = client.with_options(timeout=LIVE.timeout_s)
    classifiers: dict[str, Classifier] = {
        "decisions": DecisionsClassifier(client),
        "llm-luna": LLMProbabilityClassifier(client, MODELS.luna),
        "llm-terra": LLMProbabilityClassifier(client, MODELS.terra),
        "agent": LiveAgentClassifier(client, CONTROLS),
    }
    runs = {}
    for name, classifier in classifiers.items():
        try:
            runs[name] = await run_traced(
                name,
                _comparison_spec(name),
                list(pages)[:LIVE_PAGE_LIMIT],
                classifier,
                OpenAIRedactor(client),
                budget=budget,
                policy=DEFAULT_POLICY,
            )
        except BudgetExceededError as error:
            logger.warning("engine %s stopped by the budget: %s", name, error)
            break
    return runs


def _routes(traces: Sequence[Trace]) -> dict[str, str]:
    return {t.doc_id: t.route for t in traces}


def engine_comparison_table(runs: Mapping[str, ReviewRun]) -> pd.DataFrame:
    reference = _routes(runs["decisions"].traces)
    rows = []
    for name, run in runs.items():
        frame = traces_frame(run.traces)
        routes = _routes(run.traces)
        agree = sum(routes[d] == reference.get(d) for d in routes) / len(routes)
        cost = frame["model_cost_usd"].mean()
        rows.append(
            {
                "engine": name,
                "label": ENGINE_LABELS[name],
                "accuracy_vs_labels": frame["route_correct"].mean(),
                "agreement_vs_decisions": agree,
                "cost_per_page_usd": cost,
                "cost_per_100k_pages_usd": cost * MONTHLY_PAGES,
                "p50_latency_s": frame["latency_s"].median(),
                "mean_sync_tokens": frame["sync_tokens"].mean(),
                "escalation_rate": frame["escalated"].mean(),
                "privileged_leaks": int(frame["privileged_leak"].sum()),
                "pii_leaks": int(frame["pii_leak"].sum()),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    runs: Mapping[str, ReviewRun],
    pages: Sequence[ReviewPage],
    thresholds: Sequence[float] = SETTINGS.calibration.sweep_thresholds,
) -> pd.DataFrame:
    rows = []
    for name in TYPED_ENGINES:
        results = {
            r.doc_id: r.classification
            for r in runs[name].outcome.results
            if r.classification is not None
        }
        for threshold in thresholds:
            policy = replace(DEFAULT_POLICY, privilege_escalation=threshold)
            score = evaluate_policy(results, pages, policy)
            rows.append(
                {
                    "engine": name,
                    "privilege_threshold": threshold,
                    "escalation_rate": score.escalation_rate,
                    "privileged_leaks": score.privileged_leaks,
                }
            )
    return pd.DataFrame(rows)


async def cascade_analysis(pages: Sequence[ReviewPage] = EVAL_SET) -> pd.DataFrame:
    arms = {
        "Decisions only": await simulate_variant(ROUTER_ONLY, pages),
        "Decisions + terra cascade": await simulate_variant(DECISION_MODEL, pages),
    }
    terra = await review_pages(
        _comparison_spec("llm-terra"),
        pages,
        SimulatedTypedClassifier("llm-terra"),
        SimulatedRedactor(),
        policy=CALIBRATED_POLICY,
        mode="simulated",
    )
    arms["Terra only"] = terra.traces
    rows = []
    for arm, traces in arms.items():
        frame = traces_frame(traces)
        rows.append(
            {
                "arm": arm,
                "model_cost_per_page_usd": frame["model_cost_usd"].mean(),
                "attorney_escalations": int(frame["escalated"].sum()),
                "attorney_cost_per_page_usd": frame["attorney_cost_usd"].mean(),
                "blended_cost_per_page_usd": frame["cost_usd"].mean(),
                "mean_quality": frame["quality_score"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_engine_tradeoff(table: pd.DataFrame) -> Figure:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    sizes = 80 + 400 * table["p50_latency_s"] / table["p50_latency_s"].max()
    axis.scatter(
        table["cost_per_page_usd"], table["accuracy_vs_labels"], s=sizes, alpha=0.6
    )
    for _, row in table.iterrows():
        axis.annotate(
            row["engine"], (row["cost_per_page_usd"], row["accuracy_vs_labels"])
        )
    axis.set_xscale("log")
    axis.set_xlabel("model cost per page (USD, log scale)")
    axis.set_ylabel("route accuracy vs labels")
    axis.set_title("Engine tradeoff (marker size = p50 latency)")
    fig.tight_layout()
    return fig
