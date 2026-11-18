"""Tests for summaries, round impact and charts."""

import asyncio

import pandas as pd
import pytest
from matplotlib.figure import Figure

from doc_review_opt.reporting import (
    per_page_view,
    plot_progress,
    plot_round_chart,
    round_impact,
    summarize,
)
from doc_review_opt.simulation import simulate_all
from doc_review_opt.traces import traces_frame
from doc_review_opt.variants import BASELINE, CONTROLS, VARIANT_ORDER
from tests.helpers import at


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return traces_frame(asyncio.run(simulate_all()))


@pytest.fixture(scope="module")
def summary(frame: pd.DataFrame) -> pd.DataFrame:
    return summarize(frame)


def test_summary_has_the_cookbook_columns(summary: pd.DataFrame) -> None:
    expected = {
        "pages",
        "mean_quality",
        "policy_compliance",
        "route_accuracy",
        "escalation_accuracy",
        "escalation_rate",
        "privileged_leaks",
        "pii_leaks",
        "mean_tool_calls",
        "mean_extra_tool_calls",
        "mean_input_tokens",
        "mean_cached_tokens",
        "mean_output_tokens",
        "mean_sync_tokens",
        "mean_total_tokens",
        "p50_latency_s",
        "p95_latency_s",
        "sync_cost_per_page_usd",
        "background_cost_per_page_usd",
        "model_cost_per_page_usd",
        "attorney_cost_per_page_usd",
        "cost_per_page_usd",
        "blended_cost_per_verified_page",
        "cost_reduction_vs_baseline",
        "token_reduction_vs_baseline",
        "latency_reduction_vs_baseline",
        "monthly_cost_at_100k_pages",
    }
    assert expected <= set(summary.columns)
    assert list(summary["variant"]) == VARIANT_ORDER
    assert at(summary, 0, "cost_reduction_vs_baseline") == 0


def test_blended_cost_per_verified_page_divides_by_correct_pages(
    summary: pd.DataFrame,
) -> None:
    row = summary.iloc[0]
    assert row["blended_cost_per_verified_page"] == pytest.approx(
        row["total_cost_usd"] / row["correct_pages"]
    )
    assert row["monthly_cost_at_100k_pages"] == pytest.approx(
        row["cost_per_page_usd"] * 100_000
    )


def test_round_impact_is_the_diff_from_the_previous_round(
    summary: pd.DataFrame,
) -> None:
    impact = round_impact(summary)
    assert pd.isna(at(impact, 0, "mean_sync_tokens_delta_vs_previous"))
    assert at(impact, 1, "mean_sync_tokens_delta_vs_previous") == pytest.approx(
        at(summary, 1, "mean_sync_tokens") - at(summary, 0, "mean_sync_tokens")
    )
    assert at(impact, 1, "quality_delta_vs_previous") > 0


def test_per_page_view_filters_variants_and_columns(frame: pd.DataFrame) -> None:
    view = per_page_view(frame, [BASELINE, CONTROLS], ["route", "cost_usd"])
    assert list(view.columns) == ["doc_id", "variant_label", "route", "cost_usd"]
    assert set(view["variant_label"]) == {"Bad baseline", "Round 1: controls"}


def test_charts_return_figures_with_short_labels(summary: pd.DataFrame) -> None:
    fig = plot_round_chart(summary)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "R1: controls" in labels
    progress = plot_progress(summary, CONTROLS)
    assert len(progress.axes) == 3
    assert len(progress.axes[0].get_xticklabels()) == 2
