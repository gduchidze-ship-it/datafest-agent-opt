"""Executes the condensed demo notebook offline and checks its shape and outputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from nbclient import NotebookClient

from doc_review_opt.agent.variants import CONTROLS, VARIANT_LABELS
from scripts.build_notebook import build_demo

FLAGS = ("RUN_LIVE_API_CALLS", "RUN_LLM_JUDGE", "LANGSMITH_TRACING")
SECRETS = ("OPENAI_API_KEY", "LANGSMITH_API_KEY")
BAD_MARKERS = ("Deserializing unregistered type", "Warning")
MIN_CELLS, MAX_CELLS = 40, 75
MAX_MARKDOWN_CHARS = 700
MAX_CODE_LINES, MAX_SETUP_LINES = 12, 25
DIAGRAMS = (
    "optimization_loop",
    "bad_baseline",
    "round1",
    "round2",
    "round3",
    "round4",
    "round5",
    "human_in_the_loop",
    "decisions_or_llm",
)
INTRO_RUN = "3 pages finished, 1 waiting for an attorney: ARC-000284"


def test_demo_is_deterministic_and_sized_for_a_talk() -> None:
    first, second = build_demo(), build_demo()
    assert first.cells == second.cells
    assert MIN_CELLS <= len(first.cells) <= MAX_CELLS
    assert first.metadata["kernelspec"]["name"] == "python3"


def test_cells_stay_short() -> None:
    cells = build_demo().cells
    markdown = [c for c in cells if c.cell_type == "markdown"]
    code = [c for c in cells if c.cell_type == "code"]
    assert [c.source for c in markdown if len(c.source) > MAX_MARKDOWN_CHARS] == []
    setup, rest = code[0], code[1:]
    assert len(setup.source.splitlines()) <= MAX_SETUP_LINES
    assert [c.source for c in rest if len(c.source.splitlines()) > MAX_CODE_LINES] == []


def test_every_diagram_is_drawn() -> None:
    sources = [c.source for c in build_demo().cells if c.cell_type == "code"]
    for name in DIAGRAMS:
        assert any(f'draw_diagram("{name}")' in source for source in sources)
    assert any("draw_graph(" in source for source in sources)


def test_demo_runs_offline_without_errors_or_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for flag in FLAGS:
        monkeypatch.setenv(flag, "false")
    for secret in SECRETS:
        monkeypatch.delenv(secret, raising=False)
    notebook = NotebookClient(
        build_demo(),
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()

    outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    ]
    assert [o for o in outputs if o.output_type == "error"] == []
    streams = [o for o in outputs if o.output_type == "stream"]
    assert [o for o in streams if o.name == "stderr"] == []
    assert not [o for o in streams if any(m in o.text for m in BAD_MARKERS)]
    text = "\n".join(o.text for o in streams)
    assert f"{VARIANT_LABELS[CONTROLS]}: ACCEPTED" in text
    for expected in (
        INTRO_RUN,
        "REJECTED",
        "route ->",
        "missing_or_refused_answer",
        "Offline.",
    ):
        assert expected in text
