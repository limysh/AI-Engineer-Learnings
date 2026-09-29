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
   v
offline evaluation (Hit Rate@K, MRR)
```

## What is implemented

- deterministic cosine-similarity ranking
- explicit vector-dimension validation
- duplicate document protection
- metadata filtering before ranking
- top-k retrieval
- text-to-vector service layer for indexing and querying
- OpenAI-compatible embedding adapter with batch-response validation
- small golden-dataset evaluation using Hit Rate@K and Mean Reciprocal Rank
- unit tests that use fakes rather than network calls

## Run it

From the repository root:

```bash
python semantic_search/example.py
python -m unittest discover -s semantic_search/tests -v
```

The example and tests require no API keys and make no network calls.

For a live OpenAI embedding integration, install the OpenAI Python SDK, configure credentials using the SDK's normal environment configuration, and construct:

```python
from embeddings import OpenAIEmbeddingProvider
from service import SemanticSearchService

provider = OpenAIEmbeddingProvider.from_environment()
search = SemanticSearchService(provider)
```

No secrets are stored in this repository.

## Why this shape

Retrieval quality should be measurable independently from generation quality. If an answer is poor, I want to be able to ask whether:

1. the right documents existed in the corpus;
2. the embedding provider represented the intent well;
3. filtering removed useful candidates;
4. ranking returned the relevant documents;
5. the generator failed even though retrieval was good.

The external model client is also isolated behind a small adapter. Tests can verify batching, response ordering, dimensions, and failure handling without spending tokens or depending on a live API.

## Production extensions

For a production RAG service I would add:

- durable vector storage such as MongoDB Atlas, pgvector, or a managed vector database
- hybrid lexical + vector retrieval
- reranking
- chunk/document versioning
- per-query tracing for retrieved IDs, scores, filters, model, and latency
- a larger labelled golden dataset
- Recall@K / Precision@K / nDCG where appropriate
- retry/timeout/rate-limit policy around the embedding provider
- regression gates before changing embedding models, chunking, or retrieval parameters

## Earlier notebook

`../Semantic_Travel_Search.ipynb` is retained as the original OpenAI + MongoDB exploration. The tested code in this directory is the current reference implementation for retrieval behavior and evaluation.
