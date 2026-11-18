"""Shared test doubles: canned classifier results, stub engines and a fake redactor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from doc_review_opt.models import (
    ClassifierResult,
    PredicateAnswer,
    RedactionResult,
    RefusalAnswer,
    ReviewPage,
    ScoreAnswer,
)
from doc_review_opt.review_agent import EVAL_SET

PAGE = EVAL_SET[0]
REDACTED_TEXT = "Contact [REDACTED] at [REDACTED]."


def make_result(
    *, responsive: float, confidence: float, privileged: float, pii: float
) -> ClassifierResult:
    return ClassifierResult(
        engine="stub",
        model="stub-model",
        answers={
            "responsive": ScoreAnswer(
                name="responsive",
                score=responsive,
                confidence=confidence,
                probabilities=(0.1, 0.2, 0.7),
            ),
            "privileged": PredicateAnswer(name="privileged", probability=privileged),
            "contains_pii": PredicateAnswer(name="contains_pii", probability=pii),
        },
    )


def refused(name: str) -> RefusalAnswer:
    return RefusalAnswer(name=name)


class StubClassifier:
    def __init__(
        self,
        result: ClassifierResult | Callable[[ReviewPage], ClassifierResult],
    ) -> None:
        self._result = result
        self.seen: list[str] = []

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        self.seen.append(page.doc_id)
        if callable(self._result):
            return self._result(page)
        return self._result


class FakeRedactor:
    async def redact(self, page: ReviewPage) -> RedactionResult:
        return RedactionResult(text=REDACTED_TEXT, model="fake")


def at(frame: pd.DataFrame, row: Any, column: str) -> Any:
    return frame.loc[row, column]


def row(frame: pd.DataFrame, label: Any) -> Any:
    return frame.loc[label]
