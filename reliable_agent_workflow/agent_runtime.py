"""A small, dependency-free runtime for reliable tool execution.

The model/router is assumed to have already chosen the tool. This module owns
execution policy: review gates, retries, idempotency, state, and event history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional
import time


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class TransientToolError(RuntimeError):
    """A failure that may succeed on retry."""


class PermanentToolError(RuntimeError):
    """A failure that should not be retried."""


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    payload: Mapping[str, Any]
    idempotency_key: str
    requires_review: bool = False


@dataclass(frozen=True)
class RunEvent:
    kind: str
    detail: str
    attempt: int


@dataclass
class RunState:
    run_id: str
    status: RunStatus = RunStatus.PENDING
    attempts: int = 0
    result: Any = None
    error: Optional[str] = None
    events: List[RunEvent] = field(default_factory=list)

    def record(self, kind: str, detail: str) -> None:
        self.events.append(RunEvent(kind, detail, self.attempts))


class InMemoryStore:
    """Demo store. A production implementation should be durable."""

    def __init__(self) -> None:
        self.runs: Dict[str, RunState] = {}
        self.results_by_idempotency_key: Dict[str, Any] = {}

    def run(self, run_id: str) -> RunState:
        if run_id not in self.runs:
            self.runs[run_id] = RunState(run_id=run_id)
        return self.runs[run_id]

    def prior_result(self, key: str) -> Any:
        return self.results_by_idempotency_key.get(key)

    def remember_result(self, key: str, result: Any) -> None:
        self.results_by_idempotency_key[key] = result


class AgentRuntime:
    def __init__(
        self,
        tools: Mapping[str, Callable[[Mapping[str, Any]], Any]],
        *,
        store: Optional[InMemoryStore] = None,
        max_attempts: int = 3,
        backoff_seconds: Callable[[int], float] = lambda attempt: 0.1 * (2 ** (attempt - 1)),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.tools = dict(tools)
        self.store = store or InMemoryStore()
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.sleeper = sleeper

    def execute(
        self,
        run_id: str,
        request: ToolRequest,
        *,
        approved: bool = False,
    ) -> RunState:
        state = self.store.run(run_id)

        if request.requires_review and not approved:
            state.status = RunStatus.NEEDS_REVIEW
            state.record("review_required", "Tool execution requires human approval")
            return state

        prior = self.store.prior_result(request.idempotency_key)
        if prior is not None:
            state.status = RunStatus.SUCCEEDED
            state.result = prior
            state.record("duplicate_suppressed", "Returned result for existing idempotency key")
            return state

        tool = self.tools.get(request.tool_name)
        if tool is None:
            state.status = RunStatus.FAILED
            state.error = f"Unknown tool: {request.tool_name}"
            state.record("configuration_error", state.error)
            return state

        state.status = RunStatus.RUNNING

        while state.attempts < self.max_attempts:
            state.attempts += 1
            state.record("tool_attempt", request.tool_name)

            try:
                result = tool(request.payload)
            except TransientToolError as exc:
                state.error = str(exc)
                state.record("transient_error", state.error)

                if state.attempts >= self.max_attempts:
                    state.status = RunStatus.NEEDS_REVIEW
                    state.record("retry_budget_exhausted", "Escalating after repeated transient failures")
                    return state

                self.sleeper(self.backoff_seconds(state.attempts))
                continue
            except PermanentToolError as exc:
                state.status = RunStatus.FAILED
                state.error = str(exc)
                state.record("permanent_error", state.error)
                return state
            except Exception as exc:  # unexpected bugs are not automatically retry-safe
                state.status = RunStatus.FAILED
                state.error = f"{type(exc).__name__}: {exc}"
                state.record("unexpected_error", state.error)
                return state

            self.store.remember_result(request.idempotency_key, result)
            state.status = RunStatus.SUCCEEDED
            state.result = result
            state.error = None
            state.record("tool_succeeded", request.tool_name)
            return state

        raise AssertionError("unreachable")
