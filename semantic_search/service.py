"""Text-to-vector service layer for indexing and querying semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping

from embeddings import EmbeddingProvider
from retrieval import Document, InMemoryVectorIndex, SearchResult


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchTrace:
    query_length: int
    k: int
    metadata_filter: Mapping[str, Any]
    returned_document_ids: tuple[str, ...]
    returned_scores: tuple[float, ...]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float


class SemanticSearchService:
    """Coordinates embedding generation with deterministic vector retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        index: InMemoryVectorIndex | None = None,
        trace_sink: Callable[[SearchTrace], None] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.index = index or InMemoryVectorIndex()
        self.trace_sink = trace_sink
        self.clock = clock

    def index_documents(self, documents: list[SourceDocument]) -> None:
        if not documents:
            return

        embeddings = self.embedding_provider.embed([document.text for document in documents])
        if len(embeddings) != len(documents):
            raise RuntimeError("embedding provider returned the wrong number of vectors")

        self.index.add_many(
            [
                Document(document.document_id, document.text, tuple(embedding), document.metadata)
                for document, embedding in zip(documents, embeddings)
            ]
        )

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        started_at = self.clock()

        query_embeddings = self.embedding_provider.embed([query])
        if len(query_embeddings) != 1:
            raise RuntimeError("embedding provider must return exactly one query vector")
        embedding_finished_at = self.clock()

        filters = dict(metadata_filter or {})
        results = self.index.search(
            query_embeddings[0],
            k=k,
            metadata_filter=filters,
        )
        finished_at = self.clock()

        if self.trace_sink is not None:
            self.trace_sink(
                SearchTrace(
                    query_length=len(query),
                    k=k,
                    metadata_filter=filters,
                    returned_document_ids=tuple(result.document_id for result in results),
                    returned_scores=tuple(result.score for result in results),
                    embedding_latency_ms=(embedding_finished_at - started_at) * 1000,
                    retrieval_latency_ms=(finished_at - embedding_finished_at) * 1000,
                    total_latency_ms=(finished_at - started_at) * 1000,
                )
            )

        return results
