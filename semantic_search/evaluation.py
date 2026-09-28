"""Offline retrieval metrics for small RAG golden datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from retrieval import InMemoryVectorIndex, Vector


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    query_embedding: Vector
    relevant_document_ids: frozenset[str]
    metadata_filter: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalMetrics:
    cases: int
    hit_rate_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    index: InMemoryVectorIndex,
    cases: list[RetrievalCase],
    *,
    k: int = 3,
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not cases:
        return RetrievalMetrics(cases=0, hit_rate_at_k=0.0, mean_reciprocal_rank=0.0)

    hits = 0
    reciprocal_rank_total = 0.0

    for case in cases:
        if not case.relevant_document_ids:
            raise ValueError(f"case {case.name!r} must define at least one relevant document")

        results = index.search(
            case.query_embedding,
            k=k,
            metadata_filter=case.metadata_filter,
        )
        ranked_ids = [result.document_id for result in results]

        first_relevant_rank = next(
            (
                rank
                for rank, document_id in enumerate(ranked_ids, start=1)
                if document_id in case.relevant_document_ids
            ),
            None,
        )

        if first_relevant_rank is not None:
            hits += 1
            reciprocal_rank_total += 1.0 / first_relevant_rank

    total = len(cases)
    return RetrievalMetrics(
        cases=total,
        hit_rate_at_k=hits / total,
        mean_reciprocal_rank=reciprocal_rank_total / total,
    )
