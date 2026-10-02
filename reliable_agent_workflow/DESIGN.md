# Design Notes

## Goal

The goal is not to build another agent framework. It is to isolate a few reliability mechanisms that become easy to miss when orchestration is hidden inside a framework.

The runtime assumes that an upstream model or deterministic router has already selected a tool. It owns **execution policy**, not open-ended planning.

## 1. Explicit state over an opaque loop

A production workflow should be inspectable. Each run therefore has:

- a run ID;
- a status;
- attempt count;
- structured events;
- optional result/error.

This is intentionally boring. Boring state is easier to debug than a conversational trace that has to be reverse-engineered after an incident.

## 2. Idempotency before retries and concurrency

Retries and concurrent delivery are dangerous around side effects.

If a payment, notification, ticket creation, or database mutation times out after the remote service completed it, a blind retry can duplicate the action. A check followed later by a write is not sufficient: two workers can both observe a missing result and execute simultaneously.

The in-memory store therefore provides an atomic claim operation with three outcomes:

- **acquired**: this run owns the operation and may call the tool;
- **in progress**: another run owns it, so this run waits without executing;
- **completed**: return the stored result, including a legitimate `None` result.

Failed executions release their claim. Successful executions atomically replace the claim with the stored result.

The lock only protects threads in one Python process. A production implementation needs the same state transition in a durable shared store, typically using a unique constraint, transactional insert, or conditional write. The downstream service should also enforce the idempotency key whenever possible.

## 3. Transient vs permanent failures

Not every failure deserves a retry.

**Transient examples**
- timeout;
- temporary 429;
- brief dependency outage.

**Permanent examples**
- invalid arguments;
- permission denied;
- unsupported operation.

Retrying a permanent error adds latency and load without improving the outcome.

## 4. Human review as a state transition

For risky actions, the safest runtime behavior is often not "let the model decide harder." It is to stop and create an explicit review state.

The example also escalates repeated transient failures to review. A real system might instead use different policies per tool: DLQ, operator review, compensating action, or automatic cancellation.

## 5. Backoff

The runtime accepts a backoff function and sleeper so retry behavior is testable. Production code would typically add exponential backoff plus jitter and tool-specific retry budgets.

## 6. Observability

The event history is intentionally structured rather than free-form logging. In production, I would emit the same fields to traces/logs and measure:

- task success rate;
- retries per tool;
- terminal failure rate;
- human-review rate;
- p50/p95/p99 run latency;
- tool latency and availability;
- duplicate suppression and duplicate-in-progress counts.

## 7. What I would add for distributed execution

A real long-running runtime needs a durable scheduler/worker model:

```text
API -> durable queue -> worker -> tool
          |              |
          v              v
       run store <---- events/results
          |
          +----> review queue / UI
```

Important details would include worker leases, heartbeats, visibility timeouts, concurrency limits, backpressure, DLQ semantics, and reconciliation for stuck runs.

## 8. Why not let the LLM own this?

LLMs are useful for intent, routing, synthesis, and reasoning. They are not the right place to enforce:

- permissions;
- exactly-once-ish side-effect protection;
- retry budgets;
- schema validation;
- terminal states;
- audit history.

Those mechanisms should remain deterministic and testable.
