"""Cell constructors shared by the notebook section modules."""

from __future__ import annotations

from typing import Literal

Cell = tuple[Literal["markdown", "code"], str]


def md(text: str) -> Cell:
    return ("markdown", text.strip("\n"))


def code(text: str) -> Cell:
    return ("code", text.strip("\n"))
