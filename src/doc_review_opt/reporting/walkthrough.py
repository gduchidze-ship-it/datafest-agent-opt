"""Small typed helpers that keep each notebook round to one import."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from doc_review_opt.agent.variants import VARIANT_LABELS
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.simulation import variant_request_example
from doc_review_opt.experiment.traces import Trace, deterministic_guardrail_check
from doc_review_opt.fixtures import ATTORNEYS, EVAL_SET
from doc_review_opt.models import PageResult, ReviewPage

_TITLE_SEPARATOR = " -- "
_ATTORNEY_TAG = " (attorney)"


def _title(page: ReviewPage) -> str:
    return page.text.partition("\n")[0].split(_TITLE_SEPARATOR, 1)[-1]


def _sender(page: ReviewPage) -> str:
    return page.author + (_ATTORNEY_TAG if page.author in ATTORNEYS else "")


def page_table(
    pages: Sequence[ReviewPage], results: Sequence[PageResult] = ()
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "doc_id": [page.doc_id for page in pages],
            "page": [_title(page) for page in pages],
            "from": [_sender(page) for page in pages],
            "expected": [page.labels.expected_route for page in pages],
        }
    )
    if results:
        by_id = {result.doc_id: result for result in results}
        table["route"] = [by_id[page.doc_id].route for page in pages]
        table["decision"] = [by_id[page.doc_id].decision for page in pages]
    return table


@dataclass(frozen=True)
class GateResult:
    variant: str
    label: str
    pass_rate: float
    accepted: bool
    failures: dict[str, int]
    privileged_leaks: int
    pii_leaks: int

    def __str__(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        floor = get_settings().quality.guardrail_pass_floor
        failures = ", ".join(f"{k}={v}" for k, v in self.failures.items()) or "none"
        return (
            f"{self.label}: {verdict} (guardrail pass {self.pass_rate:.0%}, "
            f"floor {floor:.0%}; failures: {failures})"
        )


def _variant_traces(traces: Sequence[Trace], variant: str) -> list[Trace]:
    return [trace for trace in traces if trace.variant == variant]


def gate(
    traces: Sequence[Trace], variant: str, pages: Sequence[ReviewPage] = EVAL_SET
) -> GateResult:
    by_id = {page.doc_id: page for page in pages}
    chosen = _variant_traces(traces, variant)
    codes = [deterministic_guardrail_check(by_id[t.doc_id], t) for t in chosen]
    failures = Counter(code for page_codes in codes for code in page_codes)
    failed_pages = sum(1 for page_codes in codes if page_codes)
    rate = 1 - failed_pages / len(chosen)
    privileged = sum(t.privileged_leak for t in chosen)
    pii = sum(t.pii_leak for t in chosen)
    floor = get_settings().quality.guardrail_pass_floor
    return GateResult(
        variant=variant,
        label=VARIANT_LABELS[variant],
        pass_rate=rate,
        accepted=privileged == 0 and pii == 0 and rate >= floor,
        failures=dict(failures),
        privileged_leaks=privileged,
        pii_leaks=pii,
    )


def gate_table(
    traces: Sequence[Trace],
    variants: Sequence[str],
    pages: Sequence[ReviewPage] = EVAL_SET,
) -> pd.DataFrame:
    results = [gate(traces, variant, pages) for variant in variants]
    return pd.DataFrame(
        {
            "label": [r.label for r in results],
            "pass_rate": [r.pass_rate for r in results],
            "accepted": [r.accepted for r in results],
        }
    )


def _value(summary: pd.DataFrame, variant: str, column: str) -> float:
    return float(summary.loc[summary["variant"] == variant, column].iloc[0])


def symptoms(summary: pd.DataFrame, variant: str) -> str:
    def value(column: str) -> float:
        return _value(summary, variant, column)

    return "\n".join(
        (
            f"{VARIANT_LABELS[variant]}",
            f"  tool calls: {value('mean_tool_calls'):.1f} per page "
            f"({value('mean_extra_tool_calls'):.1f} unneeded)",
            f"  sync tokens: {value('mean_sync_tokens'):,.0f} per page",
            f"  p50 latency: {value('p50_latency_s'):.2f}s",
            f"  model cost: ${value('model_cost_per_page_usd'):.5f} per page",
            f"  escalation rate: {value('escalation_rate'):.0%}",
            f"  leaks: privileged {int(value('privileged_leaks'))}, "
            f"PII {int(value('pii_leaks'))}",
        )
    )


def _request_facts(request: Mapping[str, Any]) -> dict[str, object]:
    text = request.get("text", {})
    choice = request.get("tool_choice", "-")
    return {
        "model": request["model"],
        "reasoning": request.get("reasoning", {}).get("effort", "-"),
        "verbosity": text.get("verbosity", "-"),
        "structured_output": text.get("format", {}).get("name", "free text"),
        "max_output_tokens": request.get("max_output_tokens", "-"),
        "tool_choice": choice if isinstance(choice, str) else json.dumps(choice),
        "tools": ", ".join(sorted(t["name"] for t in request.get("tools", []))) or "-",
        "questions": ", ".join(q["name"] for q in request.get("questions", [])) or "-",
        "prompt_cache_key": request.get("prompt_cache_key", "-"),
        "instructions_chars": len(request.get("instructions", "")),
        "input_chars": len(json.dumps(request["input"])),
    }


def request_diff(before: str, after: str, page: ReviewPage) -> pd.DataFrame:
    old = _request_facts(variant_request_example(before, page))
    new = _request_facts(variant_request_example(after, page))
    rows = [
        {"setting": key, "before": str(old[key]), "after": str(new[key])}
        for key in old
        if old[key] != new[key]
    ]
    return pd.DataFrame(rows, columns=["setting", "before", "after"])


_METRICS = (
    ("quality", "mean_quality", "{:.2f}"),
    ("privileged leaks", "privileged_leaks", "{:d}"),
    ("PII leaks", "pii_leaks", "{:d}"),
    ("escalation rate", "escalation_rate", "{:.0%}"),
    ("tool calls per page", "mean_tool_calls", "{:.1f}"),
    ("sync tokens", "mean_sync_tokens", "{:,.0f}"),
    ("p50 latency (s)", "p50_latency_s", "{:.2f}"),
    ("model $/page", "model_cost_per_page_usd", "${:.5f}"),
    ("cost with attorney time $/page", "cost_per_page_usd", "${:.3f}"),
)
_INT_COLUMNS = ("privileged_leaks", "pii_leaks")


def before_after(summary: pd.DataFrame, previous: str, current: str) -> pd.DataFrame:
    def cell(variant: str, column: str, pattern: str) -> str:
        value = _value(summary, variant, column)
        return pattern.format(int(value) if column in _INT_COLUMNS else value)

    return pd.DataFrame(
        {
            "metric": label,
            "before": cell(previous, column, pattern),
            "after": cell(current, column, pattern),
        }
        for label, column, pattern in _METRICS
    )


__all__ = [
    "GateResult",
    "before_after",
    "gate",
    "gate_table",
    "page_table",
    "request_diff",
    "symptoms",
]
