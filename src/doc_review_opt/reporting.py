"""Per-variant summaries, round-by-round deltas, per-page views and the round charts."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from doc_review_opt.config import get_settings
from doc_review_opt.variants import BASELINE

MONTHLY_PAGES = get_settings().scoring.monthly_pages
SYNC_TOKENS_COLOR = "#4C78A8"
COST_COLOR = "#59A14F"
QUALITY_COLOR = "#E15759"
LATENCY_COLOR = "#F28E2B"
DELTA_COLUMNS = [
    "mean_sync_tokens",
    "mean_total_tokens",
    "p50_latency_s",
    "model_cost_per_page_usd",
    "cost_per_page_usd",
]
IMPACT_COLUMNS = [
    "variant_label",
    "mean_quality",
    "policy_compliance",
    "escalation_rate",
    "mean_tool_calls",
    "mean_extra_tool_calls",
    "mean_sync_tokens",
    "mean_total_tokens",
    "mean_cached_tokens",
    "p50_latency_s",
    "model_cost_per_page_usd",
    "cost_per_page_usd",
]


def summarize(traces: pd.DataFrame) -> pd.DataFrame:
    grouped = traces.groupby(["variant", "variant_label"], sort=False)
    summary = grouped.agg(
        pages=("doc_id", "count"),
        mean_quality=("quality_score", "mean"),
        policy_compliance=("policy_compliant", "mean"),
        route_accuracy=("route_correct", "mean"),
        escalation_accuracy=("escalation_correct", "mean"),
        escalation_rate=("escalated", "mean"),
        privileged_leaks=("privileged_leak", "sum"),
        pii_leaks=("pii_leak", "sum"),
        mean_tool_calls=("tool_calls", "mean"),
        mean_extra_tool_calls=("extra_tool_calls", "mean"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_cached_tokens=("cached_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        mean_sync_tokens=("sync_tokens", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        p50_latency_s=("latency_s", "median"),
        p95_latency_s=("latency_s", lambda s: s.quantile(0.95)),
        sync_cost_per_page_usd=("sync_cost_usd", "mean"),
        background_cost_per_page_usd=("background_cost_usd", "mean"),
        model_cost_per_page_usd=("model_cost_usd", "mean"),
        attorney_cost_per_page_usd=("attorney_cost_usd", "mean"),
        cost_per_page_usd=("cost_usd", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        correct_pages=("route_correct", "sum"),
    ).reset_index()
    summary["blended_cost_per_verified_page"] = summary["total_cost_usd"] / summary[
        "correct_pages"
    ].where(summary["correct_pages"] > 0)
    base = summary[summary["variant"] == BASELINE]
    base_row = base.iloc[0] if len(base) else summary.iloc[0]
    for column, source in (
        ("cost_reduction_vs_baseline", "model_cost_per_page_usd"),
        ("token_reduction_vs_baseline", "mean_total_tokens"),
        ("latency_reduction_vs_baseline", "p50_latency_s"),
    ):
        summary[column] = 1 - summary[source] / base_row[source]
    summary["monthly_cost_at_100k_pages"] = summary["cost_per_page_usd"] * MONTHLY_PAGES
    summary["monthly_model_cost_at_100k_pages"] = (
        summary["model_cost_per_page_usd"] * MONTHLY_PAGES
    )
    return summary


def round_impact(summary: pd.DataFrame) -> pd.DataFrame:
    impact = summary[IMPACT_COLUMNS].copy()
    for column in DELTA_COLUMNS:
        impact[f"{column}_delta_vs_previous"] = impact[column].diff()
    impact["quality_delta_vs_previous"] = impact["mean_quality"].diff()
    return impact


def per_page_view(
    traces: pd.DataFrame, variants: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    view = traces[traces["variant"].isin(variants)]
    return view[["doc_id", "variant_label", *columns]].reset_index(drop=True)


def _short_labels(summary: pd.DataFrame) -> pd.Series:
    return summary["variant_label"].str.replace("Round ", "R", regex=False)


def plot_round_chart(
    summary: pd.DataFrame, cost_column: str = "model_cost_per_page_usd"
) -> Figure:
    labels = _short_labels(summary)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(labels, summary["mean_sync_tokens"], color=SYNC_TOKENS_COLOR)
    axes[0].set_title("Mean synchronous tokens")
    axes[0].set_ylabel("sync tokens per page")
    axes[1].bar(labels, summary[cost_column], color=COST_COLOR)
    axes[1].set_title("Estimated cost")
    axes[1].set_ylabel("USD per page")
    axes[2].plot(labels, summary["mean_quality"], marker="o", color=QUALITY_COLOR)
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Quality score")
    axes[2].set_ylabel("score")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_progress(
    summary: pd.DataFrame,
    upto_variant: str,
    cost_column: str = "model_cost_per_page_usd",
) -> Figure:
    last = summary.index[summary["variant"] == upto_variant][0]
    shown = summary.loc[:last]
    labels = _short_labels(shown)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    for axis, column, color, title in (
        (axes[0], cost_column, COST_COLOR, "Cost per page (USD)"),
        (axes[1], "p50_latency_s", LATENCY_COLOR, "p50 latency (s)"),
        (axes[2], "mean_quality", QUALITY_COLOR, "Quality score"),
    ):
        axis.plot(labels, shown[column], marker="o", color=color)
        axis.set_title(title)
        axis.set_xlim(-0.5, len(summary) - 0.5)
        axis.tick_params(axis="x", rotation=30)
    axes[2].set_ylim(0, 1.0)
    fig.tight_layout()
    return fig
