"""Cell constructors and Mermaid helpers shared by the notebook section modules."""

from __future__ import annotations

from typing import Literal

from doc_review_opt.chart_style import PALETTE, color

Cell = tuple[Literal["markdown", "code"], str]

FILLS = {
    "produce": color("produce"),
    "withhold": color("withhold"),
    "redact": color("redact"),
    "attorney": color("attorney_review"),
    "decisions": color("decisions"),
    "before": PALETTE.grid,
    "after": PALETTE.surface,
}
STROKE_WIDTH = {"after": 3}


def md(text: str) -> Cell:
    return ("markdown", text.strip("\n"))


def code(text: str) -> Cell:
    return ("code", text.strip("\n"))


def class_def(name: str) -> str:
    width = STROKE_WIDTH.get(name, 1)
    return (
        f"    classDef {name} fill:{FILLS[name]},stroke:{PALETTE.text},"
        f"stroke-width:{width}px,color:{PALETTE.ink}"
    )


def mermaid(body: str, *classes: str) -> str:
    lines = [body.strip("\n"), *(class_def(name) for name in classes)]
    return "```mermaid\n" + "\n".join(lines) + "\n```"
