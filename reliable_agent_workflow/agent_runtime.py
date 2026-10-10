"""A small, dependency-free runtime for reliable tool execution.

The model/router is assumed to have already chosen the tool. This module owns
execution policy: review gates, retries, idempotency, state, and event history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional
import time


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ClaimStatus(str, Enum):
    ACQUIRED = "acquired"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


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


@dataclass(frozen=True)
class IdempotencyClaim:
    status: ClaimStatus
    result: Any = None
    lease_token: Optional[int] = None
    reclaimed: bool = False


@dataclass(frozen=True)
class IdempotencyLease:
    owner_id: str
    token: int
    expires_at: float


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
    """Thread-safe demo store. A production implementation should be durable."""

    def __init__(
        self,
        *,
        lease_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")

        self.runs: Dict[str, RunState] = {}
        self.results_by_idempotency_key: Dict[str, Any] = {}
        self.lease_seconds = lease_seconds
        self.clock = clock
        self._leases_by_idempotency_key: Dict[str, IdempotencyLease] = {}
        self._next_lease_token = 0
        self._lock = Lock()

    def run(self, run_id: str) -> RunState:
        with self._lock:
            if run_id not in self.runs:
                self.runs[run_id] = RunState(run_id=run_id)
            return self.runs[run_id]

    def claim_idempotency_key(self, key: str, owner_id: str) -> IdempotencyClaim:
        """Atomically acquire, recover, or inspect a side-effect lease."""
        with self._lock:
            if key in self.results_by_idempotency_key:
                return IdempotencyClaim(
                    ClaimStatus.COMPLETED,
                    self.results_by_idempotency_key[key],
                )

            now = self.clock()
            existing_lease = self._leases_by_idempotency_key.get(key)
            if existing_lease is not None and existing_lease.expires_at > now:
                return IdempotencyClaim(ClaimStatus.IN_PROGRESS)

            self._next_lease_token += 1
            lease = IdempotencyLease(
                owner_id=owner_id,
                token=self._next_lease_token,
                expires_at=now + self.lease_seconds,
            )
            self._leases_by_idempotency_key[key] = lease
            return IdempotencyClaim(
                ClaimStatus.ACQUIRED,
                lease_token=lease.token,
                reclaimed=existing_lease is not None,
            )

    def renew_idempotency_key(self, key: str, lease_token: int) -> bool:
        """Extend a lease only if the caller still owns the latest token."""
        with self._lock:
            lease = self._leases_by_idempotency_key.get(key)
            now = self.clock()
            if (
                lease is None
                or lease.token != lease_token
                or lease.expires_at <= now
            ):
                return False

            self._leases_by_idempotency_key[key] = IdempotencyLease(
                owner_id=lease.owner_id,
                token=lease.token,
                expires_at=now + self.lease_seconds,
            )
            return True

    def complete_idempotency_key(
        self,
        key: str,
        lease_token: int,
        result: Any,
    ) -> bool:
        """Commit only when the fencing token still belongs to this caller."""
        with self._lock:
            lease = self._leases_by_idempotency_key.get(key)
            if (
                lease is None
                or lease.token != lease_token
                or lease.expires_at <= self.clock()
            ):
                return False

            self.results_by_idempotency_key[key] = result
            del self._leases_by_idempotency_key[key]
            return True

    def release_idempotency_key(self, key: str, lease_token: int) -> None:
        with self._lock:
            lease = self._leases_by_idempotency_key.get(key)
            if lease is not None and lease.token == lease_token:
                del self._leases_by_idempotency_key[key]


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

        tool = self.tools.get(request.tool_name)
        if tool is None:
            state.status = RunStatus.FAILED
            state.error = f"Unknown tool: {request.tool_name}"
            state.record("configuration_error", state.error)
            return state

        claim = self.store.claim_idempotency_key(request.idempotency_key, run_id)
        if claim.status == ClaimStatus.COMPLETED:
            state.status = RunStatus.SUCCEEDED
            state.result = claim.result
            state.record("duplicate_suppressed", "Returned result for completed idempotency key")
            return state
        if claim.status == ClaimStatus.IN_PROGRESS:
            state.status = RunStatus.WAITING
            state.record("duplicate_in_progress", "Another run owns this idempotency key")
            return state

        if claim.lease_token is None:
            raise AssertionError("acquired claim must include a lease token")
        lease_token = claim.lease_token
        if claim.reclaimed:
            state.record("lease_reclaimed", "Recovered an expired idempotency lease")

        state.status = RunStatus.RUNNING
        claim_active = True

        try:
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
                        state.record(
                            "retry_budget_exhausted",
                            "Escalating after repeated transient failures",
                        )
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

                committed = self.store.complete_idempotency_key(
                    request.idempotency_key,
                    lease_token,
                    result,
                )
                if not committed:
                    state.status = RunStatus.NEEDS_REVIEW
                    state.error = "Idempotency lease was replaced before completion"
                    state.record("lease_lost", state.error)
                    return state

                claim_active = False
                state.status = RunStatus.SUCCEEDED
                state.result = result
                state.error = None
                state.record("tool_succeeded", request.tool_name)
                return state

            raise AssertionError("unreachable")
        finally:
            if claim_active:
                self.store.release_idempotency_key(
                    request.idempotency_key,
                    lease_token,
                )
