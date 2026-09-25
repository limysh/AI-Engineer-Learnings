# Reliable Agent Workflow

This is a small, dependency-free Python example of how I think about **agent reliability around tool execution**.

A model can decide *what* it wants to do. The application still needs to own whether that action is safe, retryable, observable, and idempotent.

## Why this project

The interesting part of a production agent is rarely the single LLM call. Problems show up around it:

- a tool times out after performing the side effect;
- a retry repeats the same action;
- an upstream dependency has a transient outage;
- a request is valid but high-risk and should wait for human approval;
- an execution fails three times and needs a useful escalation trail;
- an operator needs to reconstruct what happened from logs.

This runtime makes those concerns explicit.

## Flow

```text
Tool request
    |
    v
Check human-review policy ----> Needs review
    |
    v
Check idempotency cache ------> Return prior result
    |
    v
Execute tool
    |
    +---- success -------------> Persist result + succeed
    |
    +---- transient error -----> Retry with backoff
    |                              |
    |                              +--> retries exhausted -> Needs review
    |
    +---- permanent error -----> Fail without retry
```

## What is implemented

- `RunState` with explicit status and event history
- `ToolRequest` with an idempotency key
- transient and permanent tool exceptions
- bounded retries with injectable backoff/sleep
- in-memory result store for duplicate suppression
- human-review gate
- tests for retries, duplicate requests, permanent failures, and escalation

## Run it

From the repository root:

```bash
python reliable_agent_workflow/example.py
python -m unittest discover -s reliable_agent_workflow/tests -v
```

No third-party packages are required.

## Production extensions

This example intentionally stops before pretending to be a production framework. In a real distributed system I would replace or add:

- durable run state in Postgres/DynamoDB/Redis rather than memory;
- a queue between scheduling and execution;
- leases/heartbeats for long-running workers;
- DLQ or explicit terminal-failure storage;
- per-tool timeouts and circuit breakers;
- JSON-schema validation for tool inputs/outputs;
- distributed tracing and metrics by run/tool/attempt;
- RBAC and scoped credentials;
- rate limits and concurrency controls;
- durable idempotency records with retention policy;
- evaluation datasets and task-success metrics;
- a reviewer UI or workflow for escalations.

See [DESIGN.md](DESIGN.md) for the tradeoffs behind those choices.
