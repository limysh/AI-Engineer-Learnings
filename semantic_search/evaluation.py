"""Offline retrieval metrics and quality gates for RAG golden datasets."""

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


@dataclass(frozen=True)
class RetrievalThresholds:
    min_hit_rate_at_k: float
    min_mean_reciprocal_rank: float

    def __post_init__(self) -> None:
        for name, value in (
            ("min_hit_rate_at_k", self.min_hit_rate_at_k),
            ("min_mean_reciprocal_rank", self.min_mean_reciprocal_rank),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


class RetrievalQualityError(AssertionError):
    """Raised when retrieval metrics fall below an approved quality floor."""


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


def assert_retrieval_quality(
    metrics: RetrievalMetrics,
    thresholds: RetrievalThresholds,
) -> None:
    """Fail with actionable details when an offline quality floor is missed."""
    failures = []
    checks = (
        ("Hit Rate@K", metrics.hit_rate_at_k, thresholds.min_hit_rate_at_k),
        (
            "Mean Reciprocal Rank",
            metrics.mean_reciprocal_rank,
            thresholds.min_mean_reciprocal_rank,
        ),
    )

    for name, actual, minimum in checks:
        if actual < minimum:
            failures.append(f"{name} {actual:.3f} is below required {minimum:.3f}")

    if failures:
        raise RetrievalQualityError("; ".join(failures))
