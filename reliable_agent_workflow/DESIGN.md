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

## 2. Idempotency before retries

Retries are dangerous around side effects.

If a payment, notification, ticket creation, or database mutation times out after the remote service completed it, a blind retry can duplicate the action. The runtime checks a caller-provided idempotency key before executing and stores the successful result against that key.

In production, the idempotency record must be durable and ideally enforced as close to the side effect as possible.

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
- duplicate suppression count.

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
