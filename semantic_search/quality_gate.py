"""Run the checked-in semantic retrieval golden set as a CI quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation import (
    RetrievalCase,
    RetrievalMetrics,
    RetrievalThresholds,
    assert_retrieval_quality,
    evaluate_retrieval,
)
from retrieval import InMemoryVectorIndex


DEFAULT_DATASET = Path(__file__).with_name("golden_dataset.json")


def run_quality_gate(dataset_path: Path) -> RetrievalMetrics:
    dataset: dict[str, Any] = json.loads(dataset_path.read_text(encoding="utf-8"))
    index = InMemoryVectorIndex()

    for document in dataset["documents"]:
        index.add(
            document["document_id"],
            document["text"],
            document["embedding"],
            metadata=document.get("metadata"),
        )

    cases = [
        RetrievalCase(
            name=case["name"],
            query_embedding=case["query_embedding"],
            relevant_document_ids=frozenset(case["relevant_document_ids"]),
            metadata_filter=case.get("metadata_filter"),
        )
        for case in dataset["cases"]
    ]
    metrics = evaluate_retrieval(index, cases, k=int(dataset["k"]))
    configured_thresholds = dataset["thresholds"]
    thresholds = RetrievalThresholds(
        min_hit_rate_at_k=float(configured_thresholds["min_hit_rate_at_k"]),
        min_mean_reciprocal_rank=float(
            configured_thresholds["min_mean_reciprocal_rank"]
        ),
    )
    assert_retrieval_quality(metrics, thresholds)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help="path to a JSON golden dataset",
    )
    args = parser.parse_args()
    metrics = run_quality_gate(args.dataset)
    print(
        "retrieval quality gate passed: "
        f"cases={metrics.cases}, "
        f"hit_rate_at_k={metrics.hit_rate_at_k:.3f}, "
        f"mrr={metrics.mean_reciprocal_rank:.3f}"
    )


if __name__ == "__main__":
    main()
