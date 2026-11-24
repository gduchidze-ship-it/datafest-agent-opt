"""Tests for the round helpers used by the demo notebook."""

import asyncio

import pandas as pd
import pytest

from doc_review_opt.agent.graph import (
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.variants import (
    BASELINE,
    CONTROLS,
    ORIGINAL_WORDING_VARIANT,
    ROUTING,
    VARIANT_LABELS,
    VARIANT_ORDER,
    VARIANTS,
    default_policy,
)
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.simulation import build_simulated_engines, simulate_all
from doc_review_opt.experiment.traces import Trace, traces_frame
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.reporting import plot_quality_gate, summarize
from doc_review_opt.reporting.walkthrough import (
    before_after,
    gate,
    gate_table,
    page_table,
    request_diff,
    symptoms,
)

FLOOR = get_settings().quality.guardrail_pass_floor
LOW_PAGE = next(
    p for p in EVAL_SET if p.labels.risk == "low" and not p.labels.privileged
)
ROUTES = ("produce", "withhold", "redact", "attorney_review")
DEMO_PAGES = [next(p for p in EVAL_SET if p.labels.expected_route == r) for r in ROUTES]
PAGE_ROWS = [
    [
        "ARC-000141",
        "Q3 2024 REVENUE SUMMARY (INTERNAL)",
        "Finance Reporting",
        "produce",
    ],
    ["ARC-000212", "FACILITIES NOTICE", "Ines Calloway", "withhold"],
    ["ARC-000352", "SALES OPS EMAIL", "Noor Haddad", "redact"],
    ["ARC-000284", "EMAIL", "Priya Raghunathan (attorney)", "attorney_review"],
]


@pytest.fixture(scope="module")
def traces() -> list[Trace]:
    return asyncio.run(simulate_all())


@pytest.fixture(scope="module")
def all_traces() -> list[Trace]:
    return asyncio.run(simulate_all(include_extras=True))


@pytest.fixture(scope="module")
def summary(traces: list[Trace]) -> pd.DataFrame:
    return summarize(traces_frame(traces))


def test_baseline_is_rejected_and_controls_are_accepted(traces: list[Trace]) -> None:
    baseline = gate(traces, BASELINE)
    controls = gate(traces, CONTROLS)
    assert not baseline.accepted
    assert baseline.pii_leaks > 0
    assert controls.accepted
    assert str(controls).startswith(f"{VARIANT_LABELS[CONTROLS]}: ACCEPTED (")
    assert f"floor {FLOOR:.0%}" in str(controls)
    assert str(controls).endswith("failures: none)")


def test_rejected_gate_lists_failures(traces: list[Trace]) -> None:
    text = str(gate(traces, BASELINE))
    assert "REJECTED" in text
    assert "pii_in_produced_text" in text


def test_original_wording_is_rejected_on_the_floor_alone(
    all_traces: list[Trace],
) -> None:
    result = gate(all_traces, ORIGINAL_WORDING_VARIANT)
    assert not result.accepted
    assert result.privileged_leaks == 0
    assert result.pii_leaks == 0
    assert result.pass_rate == pytest.approx(28 / 30)
    assert result.pass_rate < FLOOR


def test_routing_is_accepted_with_one_missed_escalation(traces: list[Trace]) -> None:
    result = gate(traces, ROUTING)
    assert result.accepted
    assert result.pass_rate == pytest.approx(29 / 30)
    assert result.failures == {"escalation_mismatch": 1}


def test_gate_table_feeds_the_quality_chart(traces: list[Trace]) -> None:
    table = gate_table(traces, VARIANT_ORDER)
    assert list(table.columns) == ["label", "pass_rate", "accepted"]
    assert list(table["label"]) == [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    assert plot_quality_gate(table, FLOOR) is not None


def test_symptoms_show_the_published_baseline_numbers(summary: pd.DataFrame) -> None:
    text = symptoms(summary, BASELINE)
    for fragment in (
        "5.0 per page (4.2 unneeded)",
        "8,287",
        "3.59s",
        "$0.00879",
        "50%",
        "privileged 0, PII 2",
    ):
        assert fragment in text
    assert "guardrail failures" not in text


def test_request_diff_shows_only_changed_settings() -> None:
    diff = request_diff(BASELINE, CONTROLS, LOW_PAGE)
    assert list(diff.columns) == ["setting", "before", "after"]
    assert "structured_output" in set(diff["setting"])
    assert (diff["before"] != diff["after"]).all()


def test_request_diff_is_empty_when_nothing_differs() -> None:
    assert request_diff(CONTROLS, CONTROLS, LOW_PAGE).empty


def test_request_diff_shows_the_model_change_for_routing() -> None:
    diff = request_diff(CONTROLS, ROUTING, LOW_PAGE)
    assert "model" in set(diff["setting"])


def test_before_after_has_one_formatted_row_per_metric(summary: pd.DataFrame) -> None:
    table = before_after(summary, BASELINE, CONTROLS)
    assert list(table.columns) == ["metric", "before", "after"]
    assert len(table) == 9
    before = dict(zip(table["metric"].tolist(), table["before"].tolist(), strict=True))
    after = dict(zip(table["metric"].tolist(), table["after"].tolist(), strict=True))
    assert before["privileged leaks"] == "0"
    assert before["PII leaks"] == "2"
    assert after["PII leaks"] == "0"
    assert before["model $/page"] == "$0.00879"
    assert after["model $/page"] == "$0.00131"
    assert after["model $/page"].startswith("$0.")
    assert after["escalation rate"].endswith("%")


def test_page_table_shows_each_page_in_order() -> None:
    table = page_table(DEMO_PAGES)
    assert list(table.columns) == ["doc_id", "page", "from", "expected"]
    assert table.to_numpy().tolist() == PAGE_ROWS


def test_page_table_drops_only_the_leading_title_prefix() -> None:
    copy = next(p for p in EVAL_SET if p.doc_id == "ARC-000572")
    assert page_table([copy])["page"].tolist() == [
        "Q3 2024 REVENUE SUMMARY (INTERNAL) -- COPY"
    ]


async def test_page_table_adds_route_and_decision_after_a_resumed_run() -> None:
    policy = default_policy(VARIANTS[BASELINE])
    engine, redactor = build_simulated_engines(VARIANTS[BASELINE], EVAL_SET, policy)
    graph = build_review_graph(engine, redactor, review_checkpointer(), policy=policy)
    outcome = await run_batch(
        graph, DEMO_PAGES, thread_id="page-table", attorney_decisions={}
    )
    assert [p["doc_id"] for p in outcome.interrupts] == ["ARC-000284"]
    table = page_table(DEMO_PAGES, outcome.results[::-1])
    assert table[["doc_id", "page", "from", "expected"]].to_numpy().tolist() == (
        PAGE_ROWS
    )
    assert list(table["route"]) == list(ROUTES)
    assert list(table["decision"]) == ["produce", "withhold", "produce", "withhold"]
