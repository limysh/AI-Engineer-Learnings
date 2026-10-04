# Semantic Retrieval + Evaluation

A small, provider-agnostic retrieval core for reasoning about the part of RAG that is easiest to hand-wave and hardest to debug: **what context did we retrieve, why did it rank that way, and how do we know retrieval improved?**

The code separates embedding generation from ranking. That keeps retrieval deterministic and regression-testable while still allowing a real embedding provider to be plugged in at the service boundary.

## Flow

```text
documents
   |
embedding provider
   |
   v
vector index ---- metadata filters
   |
query text
   |
embedding provider
   |
   v
cosine ranking
   |
top-k context
   |
   +----> privacy-aware trace (latency, filters, result IDs/scores)
   |
   v
offline evaluation (Hit Rate@K, MRR)
   |
   v
CI quality gate
```

For hybrid retrieval, the same filtered corpus is ranked independently by cosine similarity and BM25, then combined with reciprocal-rank fusion (RRF). This lets exact product names, identifiers, and domain terms complement semantic similarity without making the two score scales directly comparable.

## What is implemented

- deterministic cosine-similarity ranking
- explicit vector-dimension validation
- duplicate document protection
- metadata filtering before ranking
- top-k retrieval
- dependency-free BM25 lexical ranking
- hybrid lexical + vector retrieval using reciprocal-rank fusion
- text-to-vector service layer for indexing and querying
- OpenAI-compatible embedding adapter with batch-response validation
- privacy-aware structured search traces with embedding, retrieval, and total latency
- result ID/score telemetry without storing raw query text
- small golden-dataset evaluation using Hit Rate@K and Mean Reciprocal Rank
- configurable retrieval quality floors with actionable failures
- unit tests that use fakes rather than network calls

## Run it

From the repository root:

```bash
python semantic_search/example.py
python -m unittest discover -s semantic_search/tests -v
python semantic_search/quality_gate.py
```

The example, tests, and quality gate require no API keys and make no network calls.

For a live OpenAI embedding integration, install the OpenAI Python SDK, configure credentials using the SDK's normal environment configuration, and construct:

```python
from embeddings import OpenAIEmbeddingProvider
from service import SemanticSearchService

provider = OpenAIEmbeddingProvider.from_environment()
search = SemanticSearchService(provider)
```

No secrets are stored in this repository.

## Retrieval quality gate

`golden_dataset.json` is a small, version-controlled evaluation fixture. It contains the corpus embeddings, labelled queries, metadata filters, `k`, and approved minimum scores. `quality_gate.py` rebuilds the index, evaluates every case, and exits non-zero when Hit Rate@K or Mean Reciprocal Rank drops below those floors.

GitHub Actions runs the gate separately from unit tests so a pull request cannot silently change ranking, filtering, embeddings, or evaluation data and reduce the approved retrieval quality. Changes to the golden set or its thresholds should be reviewed as explicit quality decisions, not incidental test updates.

## Hybrid retrieval

`hybrid.py` addresses a common vector-search weakness: exact terms can be missed when an embedding ranks a semantically nearby document higher. It computes BM25 and cosine rankings independently and combines their rank positions with RRF:

```text
query text --------> BM25 rank ---+
                                  +--> RRF --> top-k
query embedding --> cosine rank --+
```

RRF uses rank positions rather than raw scores, avoiding brittle normalization between BM25 and cosine similarity. Metadata filtering is applied before both rankings, and queries with no lexical overlap naturally fall back to vector order.

## Why this shape

Retrieval quality should be measurable independently from generation quality. If an answer is poor, I want to be able to ask whether:

1. the right documents existed in the corpus;
2. the embedding provider represented the intent well;
3. filtering removed useful candidates;
4. ranking returned the relevant documents;
5. the generator failed even though retrieval was good.

The external model client is isolated behind a small adapter. Tests can verify batching, response ordering, dimensions, and failure handling without spending tokens or depending on a live API.

The search service also exposes an optional trace sink. Successful searches emit structured timing and retrieval metadata, but deliberately omit the raw query. In a real enterprise system, query text can contain customer or sensitive data, so observability should be useful without making logs an unnecessary second copy of user content.

## Production extensions

For a production RAG service I would add:

- durable vector storage such as MongoDB Atlas, pgvector, or a managed vector database
- reranking
- chunk/document versioning
- trace correlation IDs and failure/error telemetry
- a larger labelled golden dataset built from representative traffic
- Recall@K / Precision@K / nDCG where appropriate
- retry/timeout/rate-limit policy around the embedding provider
- baseline-delta gates when embedding models, chunking, or retrieval parameters change

## Earlier notebook

`../Semantic_Travel_Search.ipynb` is retained as the original OpenAI + MongoDB exploration. The tested code in this directory is the current reference implementation for retrieval behavior and evaluation.
