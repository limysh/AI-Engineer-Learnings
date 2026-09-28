# Semantic Retrieval + Evaluation

A small, provider-agnostic retrieval core for reasoning about the part of RAG that is easiest to hand-wave and hardest to debug: **what context did we retrieve, why did it rank that way, and how do we know retrieval improved?**

The code deliberately separates embedding generation from ranking. That keeps the retrieval logic deterministic and makes it possible to test quality without calling an external model on every test run.

## Flow

```text
documents
   |
embedding provider
   |
   v
vector index ---- metadata filters
   |
query embedding
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
- small golden-dataset evaluation using Hit Rate@K and Mean Reciprocal Rank
- unit tests for ranking, filtering, validation, duplicates, and metrics

## Run it

From the repository root:

```bash
python semantic_search/example.py
python -m unittest discover -s semantic_search/tests -v
```

No third-party packages or API keys are required for the example or tests.

The example uses hand-authored toy vectors so the behavior is deterministic. In a real system, the vectors would come from an embedding provider and the index would typically live in a vector database.

## Why this shape

Retrieval quality should be measurable independently from generation quality. If an answer is poor, I want to be able to ask whether:

1. the right documents existed in the corpus;
2. the query embedding represented the intent well;
3. filtering removed useful candidates;
4. ranking returned the relevant documents;
5. the generator failed even though retrieval was good.

Keeping retrieval and evaluation explicit makes those failure modes easier to isolate.

## Production extensions

For a production RAG service I would add:

- OpenAI/Azure OpenAI or another embedding-provider adapter
- durable vector storage such as MongoDB Atlas, pgvector, or a managed vector database
- hybrid lexical + vector retrieval
- reranking
- chunk/document versioning
- per-query tracing for retrieved IDs, scores, filters, and latency
- a larger labelled golden dataset
- Recall@K / Precision@K / nDCG where appropriate
- regression gates before changing embedding models, chunking, or retrieval parameters

## Earlier notebook

`../Semantic_Travel_Search.ipynb` is retained as the original OpenAI + MongoDB exploration. The tested code in this directory is the cleaner reference implementation for retrieval behavior and evaluation.
