"""Compare vector and hybrid retrieval on labelled, deterministic RAG queries."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from evaluation import (
    RetrievalMetrics,
    RetrievalQualityError,
    RetrievalThresholds,
    assert_retrieval_quality,
)
from hybrid import HybridSearchIndex
from retrieval import InMemoryVectorIndex


DEFAULT_DATASET = Path(__file__).with_name("hybrid_golden_dataset.json")


@dataclass(frozen=True)
class HybridComparison:
    vector: RetrievalMetrics
    hybrid: RetrievalMetrics
    rescued_cases: tuple[str, ...]


def _first_relevant_rank(results, relevant_ids: set[str]) -> int | None:
    return next(
        (rank for rank, result in enumerate(results, 1)
         if result.document_id in relevant_ids),
        None,
    )


def _metrics(ranks: list[int | None]) -> RetrievalMetrics:
    count = len(ranks)
    return RetrievalMetrics(
        cases=count,
        hit_rate_at_k=sum(rank is not None for rank in ranks) / count,
        mean_reciprocal_rank=sum(1.0 / rank for rank in ranks if rank is not None) / count,
    )


def run_hybrid_quality_gate(dataset_path: Path = DEFAULT_DATASET) -> HybridComparison:
    """Gate hybrid quality and catch regressions against the paired vector baseline."""
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    k = int(dataset["k"])
    if k < 1:
        raise ValueError("k must be >= 1")
    if not dataset["cases"]:
        raise ValueError("hybrid golden dataset must contain at least one case")

    vector_index = InMemoryVectorIndex()
    hybrid_index = HybridSearchIndex()
    for document in dataset["documents"]:
        fields = (
            document["document_id"], document["text"], document["embedding"]
        )
        metadata = document.get("metadata")
        vector_index.add(*fields, metadata=metadata)
        hybrid_index.add(*fields, metadata=metadata)

    vector_ranks: list[int | None] = []
    hybrid_ranks: list[int | None] = []
    rescued: list[str] = []
    for case in dataset["cases"]:
        relevant_ids = set(case["relevant_document_ids"])
        if not relevant_ids:
            raise ValueError(f"case {case['name']!r} has no relevant document IDs")
        options = {"k": k, "metadata_filter": case.get("metadata_filter")}
        vector_rank = _first_relevant_rank(
            vector_index.search(case["query_embedding"], **options), relevant_ids
        )
        hybrid_rank = _first_relevant_rank(
            hybrid_index.search(case["query_text"], case["query_embedding"], **options),
            relevant_ids,
        )
        vector_ranks.append(vector_rank)
        hybrid_ranks.append(hybrid_rank)
        if vector_rank is None and hybrid_rank is not None:
            rescued.append(case["name"])

    comparison = HybridComparison(
        vector=_metrics(vector_ranks),
        hybrid=_metrics(hybrid_ranks),
        rescued_cases=tuple(rescued),
    )
    configured = dataset["thresholds"]
    assert_retrieval_quality(
        comparison.hybrid,
        RetrievalThresholds(
            min_hit_rate_at_k=float(configured["min_hit_rate_at_k"]),
            min_mean_reciprocal_rank=float(configured["min_mean_reciprocal_rank"]),
        ),
    )
    if (
        comparison.hybrid.hit_rate_at_k + 1e-12 < comparison.vector.hit_rate_at_k
        or comparison.hybrid.mean_reciprocal_rank + 1e-12
        < comparison.vector.mean_reciprocal_rank
    ):
        raise RetrievalQualityError(
            "Hybrid retrieval regressed below vector-only baseline: "
            f"vector hit={comparison.vector.hit_rate_at_k:.3f}, "
            f"MRR={comparison.vector.mean_reciprocal_rank:.3f}; "
            f"hybrid hit={comparison.hybrid.hit_rate_at_k:.3f}, "
            f"MRR={comparison.hybrid.mean_reciprocal_rank:.3f}"
        )
    minimum_rescued = int(configured.get("min_rescued_cases", 1))
    if minimum_rescued < 0:
        raise ValueError("min_rescued_cases must be >= 0")
    if len(rescued) < minimum_rescued:
        raise RetrievalQualityError(
            f"Hybrid rescued {len(rescued)} cases; required {minimum_rescued}"
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    result = run_hybrid_quality_gate(args.dataset)
    print(
        "hybrid quality gate passed: "
        f"cases={result.hybrid.cases}, "
        f"vector_hit_rate_at_k={result.vector.hit_rate_at_k:.3f}, "
        f"vector_mrr={result.vector.mean_reciprocal_rank:.3f}, "
        f"hybrid_hit_rate_at_k={result.hybrid.hit_rate_at_k:.3f}, "
        f"hybrid_mrr={result.hybrid.mean_reciprocal_rank:.3f}, "
        f"rescued={len(result.rescued_cases)}"
    )


if __name__ == "__main__":
    main()
