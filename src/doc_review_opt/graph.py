"""The review graph: classify each page, route it, and finish in one terminal node."""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
)

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, interrupt
from langsmith import traceable
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from doc_review_opt import models as review_models
from doc_review_opt._responses import completed_text, usage_from_response
from doc_review_opt.config import get_settings
from doc_review_opt.decisions import DEFAULT_POLICY, RoutingPolicy, decide, flags_pii
from doc_review_opt.exceptions import BilledCallError, UnusableReplyError
from doc_review_opt.models import (
    BilledSpend,
    ClassifierResult,
    Disposition,
    PageResult,
    PredicateAnswer,
    RedactionResult,
    ReviewPage,
    Route,
    ScoreAnswer,
)
from doc_review_opt.pricing import estimate_cost_usd
from doc_review_opt.review_agent import PROTOCOL, REDACTION_PROMPT

logger = logging.getLogger(__name__)

SETTINGS = get_settings()
REDACTION_MODEL = SETTINGS.models.redaction
ATTORNEY_DECISIONS = ("produce", "withhold")
InterruptPayload = dict[str, Any]
ReviewGraph = CompiledStateGraph[Any, Any, Any, Any]


STATE_MODELS = [
    (review_models.__name__, name)
    for name, value in vars(review_models).items()
    if isinstance(value, type)
    and issubclass(value, BaseModel)
    and value.__module__ == review_models.__name__
]


def review_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS))


class InvalidAttorneyDecisionError(ValueError): ...


class RedactionError(BilledCallError): ...


class Classifier(Protocol):
    async def classify(
        self, page: ReviewPage, protocol: str, /
    ) -> ClassifierResult: ...


class Redactor(Protocol):
    async def redact(self, page: ReviewPage) -> RedactionResult: ...


class OpenAIRedactor:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client
        self._traced = traceable(
            run_type="llm",
            name="redact_page",
            metadata={"engine": "redactor", "model": REDACTION_MODEL},
        )(self._redact)

    async def redact(self, page: ReviewPage) -> RedactionResult:
        result: RedactionResult = await self._traced(page.text)
        return result

    async def _redact(self, text: str) -> RedactionResult:
        started = perf_counter()
        try:
            response = await self._client.responses.create(
                model=REDACTION_MODEL,
                input=REDACTION_PROMPT + text,
                reasoning={"effort": SETTINGS.live.reasoning_effort},
                text={"verbosity": "low"},
                max_output_tokens=SETTINGS.live.redaction_max_output_tokens,
            )
        except OpenAIError as error:
            raise RedactionError(str(error)) from error
        usage = usage_from_response(response)
        try:
            redacted = completed_text(response)
        except UnusableReplyError as error:
            cost = estimate_cost_usd(REDACTION_MODEL, usage)
            raise RedactionError(str(error), usage, cost) from error
        return RedactionResult(
            text=redacted,
            model=REDACTION_MODEL,
            usage=usage,
            latency_s=perf_counter() - started,
            cost_usd=estimate_cost_usd(REDACTION_MODEL, usage),
        )


class PageState(TypedDict):
    doc_id: str
    page: ReviewPage
    protocol: str
    classification: ClassifierResult
    route: Route
    reason: str
    failed_redaction: NotRequired[BilledSpend]
    results: Annotated[list[PageResult], operator.add]


class BatchState(TypedDict):
    pages: list[ReviewPage]
    results: Annotated[list[PageResult], operator.add]


class _AttorneyText(NamedTuple):
    text: str | None
    redaction: RedactionResult | None
    decision: Literal["produce", "withhold"]
    failed_redaction: BilledSpend


def _spent_by(error: RedactionError) -> BilledSpend:
    return BilledSpend(usage=error.usage, cost_usd=error.cost_usd)


def interrupt_payload(
    doc_id: str, reason: str, classification: ClassifierResult
) -> InterruptPayload:
    privileged = classification.answers.get("privileged")
    responsive = classification.answers.get("responsive")
    return {
        "doc_id": doc_id,
        "reason": reason,
        "privileged": (
            privileged.probability if isinstance(privileged, PredicateAnswer) else None
        ),
        "responsive_score": (
            responsive.score if isinstance(responsive, ScoreAnswer) else None
        ),
        "responsive_confidence": (
            responsive.confidence if isinstance(responsive, ScoreAnswer) else None
        ),
    }


class _PageNodes:
    def __init__(
        self, classifier: Classifier, redactor: Redactor, policy: RoutingPolicy
    ) -> None:
        self._classifier = classifier
        self._redactor = redactor
        self._policy = policy

    async def classify(self, state: PageState) -> dict[str, Any]:
        result = await self._classifier.classify(state["page"], state["protocol"])
        decision = decide(result, self._policy)
        return {
            "classification": result,
            "route": decision.route,
            "reason": decision.reason,
        }

    async def attorney_review(self, state: PageState) -> dict[str, Any]:
        payload = interrupt_payload(
            state["doc_id"], state["reason"], state["classification"]
        )
        decision = interrupt(payload)
        if decision not in ATTORNEY_DECISIONS:
            raise InvalidAttorneyDecisionError(
                f"{state['doc_id']}: {decision!r} is not one of {ATTORNEY_DECISIONS}"
            )
        produced = await self._attorney_text(state, decision)
        result = PageResult(
            doc_id=state["doc_id"],
            decision=produced.decision,
            route="attorney_review",
            redacted=produced.redaction is not None,
            produced_text=produced.text,
            escalated=True,
            attorney_decision=decision,
            classification=state["classification"],
            redaction=produced.redaction,
            failed_redaction=produced.failed_redaction,
            reason=state["reason"],
        )
        return {"results": [result]}

    async def _attorney_text(self, state: PageState, decision: str) -> _AttorneyText:
        earlier = state.get("failed_redaction", BilledSpend())
        if decision != "produce":
            return _AttorneyText(None, None, "withhold", earlier)
        page = state["page"]
        if not flags_pii(state["classification"], self._policy):
            return _AttorneyText(page.text, None, "produce", earlier)
        try:
            redaction = await self._redactor.redact(page)
        except RedactionError as error:
            logger.warning(
                "redaction failed doc_id=%s error=%s", state["doc_id"], error
            )
            return _AttorneyText(None, None, "withhold", earlier + _spent_by(error))
        return _AttorneyText(redaction.text, redaction, "produce", earlier)

    async def redact(
        self, state: PageState
    ) -> Command[Literal["attorney_review"]] | dict[str, Any]:
        try:
            redaction = await self._redactor.redact(state["page"])
        except RedactionError as error:
            logger.warning(
                "redaction failed doc_id=%s error=%s", state["doc_id"], error
            )
            return Command(
                goto="attorney_review",
                update={
                    "route": "attorney_review",
                    "reason": "redaction_failed",
                    "failed_redaction": _spent_by(error),
                },
            )
        result = PageResult(
            doc_id=state["doc_id"],
            decision="produce",
            route="redact",
            redacted=True,
            produced_text=redaction.text,
            classification=state["classification"],
            redaction=redaction,
            reason=state["reason"],
        )
        return {"results": [result]}

    async def produce(self, state: PageState) -> dict[str, Any]:
        result = PageResult(
            doc_id=state["doc_id"],
            decision="produce",
            route="produce",
            produced_text=state["page"].text,
            classification=state["classification"],
            reason=state["reason"],
        )
        return {"results": [result]}

    async def withhold(self, state: PageState) -> dict[str, Any]:
        result = PageResult(
            doc_id=state["doc_id"],
            decision="withhold",
            route="withhold",
            classification=state["classification"],
            reason=state["reason"],
        )
        return {"results": [result]}


def _pick_route(state: PageState) -> Route:
    return state["route"]


def _compile_page_graph(nodes: _PageNodes) -> Any:
    page_graph = StateGraph(PageState)
    page_graph.add_node("classify", nodes.classify)
    page_graph.add_node("attorney_review", nodes.attorney_review)
    page_graph.add_node("redact", nodes.redact)
    page_graph.add_node("produce", nodes.produce)
    page_graph.add_node("withhold", nodes.withhold)
    page_graph.add_edge(START, "classify")
    page_graph.add_conditional_edges(
        "classify", _pick_route, ["attorney_review", "redact", "produce", "withhold"]
    )
    for node in ("attorney_review", "redact", "produce", "withhold"):
        page_graph.add_edge(node, END)
    return page_graph.compile()


def build_review_graph(
    classifier: Classifier,
    redactor: Redactor,
    checkpointer: BaseCheckpointSaver[Any],
    *,
    policy: RoutingPolicy = DEFAULT_POLICY,
    protocol: str = PROTOCOL,
) -> ReviewGraph:

    def fan_out(state: BatchState) -> list[Send]:
        return [
            Send(
                "review_page",
                {"doc_id": page.doc_id, "page": page, "protocol": protocol},
            )
            for page in state["pages"]
        ]

    batch = StateGraph(BatchState)
    batch.add_node(
        "review_page", _compile_page_graph(_PageNodes(classifier, redactor, policy))
    )
    batch.add_conditional_edges(START, fan_out, ["review_page"])
    batch.add_edge("review_page", END)
    return batch.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class BatchOutcome:
    results: tuple[PageResult, ...]
    interrupts: tuple[InterruptPayload, ...]


async def run_batch(
    graph: ReviewGraph,
    pages: Sequence[ReviewPage],
    *,
    thread_id: str,
    attorney_decisions: Mapping[str, Disposition],
    default: Literal["produce", "withhold"] = "withhold",
    max_concurrency: int | None = None,
) -> BatchOutcome:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    state = await graph.ainvoke({"pages": list(pages), "results": []}, config)
    payloads: list[InterruptPayload] = []
    while state.get("__interrupt__"):
        resume: dict[str, Disposition] = {}
        for pending in state["__interrupt__"]:
            payloads.append(pending.value)
            resume[pending.id] = attorney_decisions.get(
                pending.value["doc_id"], default
            )
        state = await graph.ainvoke(Command(resume=resume), config)
    return BatchOutcome(
        results=tuple(cast(list[PageResult], state["results"])),
        interrupts=tuple(payloads),
    )
