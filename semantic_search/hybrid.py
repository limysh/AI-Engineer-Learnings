"""Dependency-free hybrid lexical and vector retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
import re
from typing import Any, Mapping

from retrieval import InMemoryVectorIndex, Vector


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class HybridDocument:
    document_id: str
    text: str
    metadata: Mapping[str, Any]
    term_counts: Counter[str]


@dataclass(frozen=True)
class HybridSearchResult:
    document_id: str
    text: str
    score: float
    vector_rank: int
    lexical_rank: int | None
    metadata: Mapping[str, Any]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class HybridSearchIndex:
    """Combines cosine and BM25 rankings using reciprocal-rank fusion."""

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        if bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be > 0")
        if not 0.0 <= bm25_b <= 1.0:
            raise ValueError("bm25_b must be between 0 and 1")

        self.rrf_k = rrf_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._vector_index = InMemoryVectorIndex()
        self._documents: list[HybridDocument] = []

    def add(
        self,
        document_id: str,
        text: str,
        embedding: Vector,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        document_metadata = dict(metadata or {})
        self._vector_index.add(
            document_id,
            text,
            embedding,
            metadata=document_metadata,
        )
        self._documents.append(
            HybridDocument(
                document_id=document_id,
                text=text,
                metadata=document_metadata,
                term_counts=Counter(tokenize(text)),
            )
        )

    def search(
        self,
        query_text: str,
        query_embedding: Vector,
        *,
        k: int = 3,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if k < 1:
            raise ValueError("k must be >= 1")

        filters = dict(metadata_filter or {})
        candidates = [
            document
            for document in self._documents
            if all(document.metadata.get(key) == value for key, value in filters.items())
        ]
        if not candidates:
            return []

        vector_results = self._vector_index.search(
            query_embedding,
            k=len(candidates),
            metadata_filter=filters,
        )
        vector_rank_by_id = {
            result.document_id: rank
            for rank, result in enumerate(vector_results, start=1)
        }

        lexical_scores = self._bm25_scores(tokenize(query_text), candidates)
        lexical_ranking = sorted(
            (
                (document_id, score)
                for document_id, score in lexical_scores.items()
                if score > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        lexical_rank_by_id = {
            document_id: rank
            for rank, (document_id, _) in enumerate(lexical_ranking, start=1)
        }

        fused_results = []
        for document in candidates:
            vector_rank = vector_rank_by_id[document.document_id]
            lexical_rank = lexical_rank_by_id.get(document.document_id)
            score = 1.0 / (self.rrf_k + vector_rank)
            if lexical_rank is not None:
                score += 1.0 / (self.rrf_k + lexical_rank)

            fused_results.append(
                HybridSearchResult(
                    document_id=document.document_id,
                    text=document.text,
                    score=score,
                    vector_rank=vector_rank,
                    lexical_rank=lexical_rank,
                    metadata=document.metadata,
                )
            )

        return sorted(
            fused_results,
            key=lambda result: (-result.score, result.document_id),
        )[:k]

    def _bm25_scores(
        self,
        query_tokens: list[str],
        candidates: list[HybridDocument],
    ) -> dict[str, float]:
        scores = {document.document_id: 0.0 for document in candidates}
        if not query_tokens:
            return scores

        document_count = len(candidates)
        average_length = sum(
            sum(document.term_counts.values()) for document in candidates
        ) / document_count
        if average_length == 0:
            return scores

        for term in set(query_tokens):
            document_frequency = sum(
                1 for document in candidates if document.term_counts[term] > 0
            )
            if document_frequency == 0:
                continue

            inverse_document_frequency = log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for document in candidates:
                term_frequency = document.term_counts[term]
                if term_frequency == 0:
                    continue

                document_length = sum(document.term_counts.values())
                normalization = term_frequency + self.bm25_k1 * (
                    1.0
                    - self.bm25_b
                    + self.bm25_b * document_length / average_length
                )
                scores[document.document_id] += (
                    inverse_document_frequency
                    * term_frequency
                    * (self.bm25_k1 + 1.0)
                    / normalization
                )

        return scores
