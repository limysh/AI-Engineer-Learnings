"""Small, dependency-free vector retrieval core used for RAG experiments.

Embedding generation is intentionally kept outside this module. Production systems
can plug in OpenAI, Azure OpenAI, Cohere, local models, or precomputed vectors
without changing the ranking and evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Mapping, Sequence


Vector = Sequence[float]


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    embedding: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    text: str
    score: float
    metadata: Mapping[str, Any]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    if not left:
        raise ValueError("vectors must not be empty")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


class InMemoryVectorIndex:
    """Deterministic reference index for retrieval experiments and evaluation."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._dimension: int | None = None
        self._ids: set[str] = set()

    def add(
        self,
        document_id: str,
        text: str,
        embedding: Vector,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        vector = tuple(float(value) for value in embedding)
        if not document_id:
            raise ValueError("document_id must not be empty")
        if document_id in self._ids:
            raise ValueError(f"duplicate document_id: {document_id}")
        if not vector:
            raise ValueError("embedding must not be empty")

        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise ValueError(
                f"embedding dimension {len(vector)} does not match index dimension {self._dimension}"
            )

        self._documents.append(
            Document(
                document_id=document_id,
                text=text,
                embedding=vector,
                metadata=dict(metadata or {}),
            )
        )
        self._ids.add(document_id)

    def search(
        self,
        query_embedding: Vector,
        *,
        k: int = 3,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._dimension is None:
            return []

        query = tuple(float(value) for value in query_embedding)
        if len(query) != self._dimension:
            raise ValueError(
                f"query dimension {len(query)} does not match index dimension {self._dimension}"
            )

        filters = dict(metadata_filter or {})
        candidates = [
            document
            for document in self._documents
            if all(document.metadata.get(key) == value for key, value in filters.items())
        ]

        ranked = sorted(
            (
                SearchResult(
                    document_id=document.document_id,
                    text=document.text,
                    score=cosine_similarity(query, document.embedding),
                    metadata=document.metadata,
                )
                for document in candidates
            ),
            key=lambda result: (-result.score, result.document_id),
        )
        return ranked[:k]
