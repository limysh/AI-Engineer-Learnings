# AI Engineer Learnings

A small collection of hands-on AI engineering projects and design exercises focused on the part of AI work I care about most: **making probabilistic systems reliable enough to use in real software**.

This repository is intentionally separate from employer/client code and uses only synthetic or public examples.

## Projects

### 1. Reliable Agent Workflow
A compact Python runtime that explores production concerns around tool-using agents:

- explicit run state rather than a hidden agent loop
- retries for transient failures
- idempotency for duplicate requests
- permanent vs transient error classification
- human-review fallback for risky or repeatedly failing actions
- structured event history for debugging and observability
- unit tests for the failure paths, not only the happy path

**Start here:** [reliable_agent_workflow/README.md](reliable_agent_workflow/README.md)

The project is deliberately small. The goal is to make the control flow and tradeoffs easy to inspect rather than hide them behind a large framework.

### 2. Semantic Travel Search
A notebook exploring the retrieval side of a RAG-style system using OpenAI embeddings and MongoDB.

It covers:

- embedding unstructured text
- storing searchable records in MongoDB
- the shape of a semantic retrieval workflow
- how a retrieval component can become the grounding layer for a larger RAG system

**Notebook:** [Semantic_Travel_Search.ipynb](Semantic_Travel_Search.ipynb)

## Engineering principles I am exploring

1. **Keep deterministic work deterministic.** LLMs are useful for interpretation and reasoning; permissions, validation, retries, and side-effect control belong in application code.
2. **Design for failure before adding autonomy.** Tool calls fail, model outputs vary, and retries can duplicate side effects.
3. **Evaluation is part of the system.** A workflow is not finished when it produces output; it needs observable success criteria and regression tests.
4. **Human review is a feature, not an embarrassment.** High-risk or ambiguous actions should have an explicit escalation path.
5. **Production AI is mostly systems engineering around the model.**

## Tech
Python, OpenAI APIs, MongoDB, embeddings, agent/tool orchestration patterns, testing, reliability, and evaluation.

## About
I am a senior software/AI engineer with a backend and distributed-systems background. I use this repository for small experiments that help me reason about production AI architecture, not as a dump of proprietary work.
