"""Text-to-vector service layer for indexing and querying semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from embeddings import EmbeddingProvider
from retrieval import InMemoryVectorIndex, SearchResult


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SemanticSearchService:
    """Coordinates embedding generation with deterministic vector retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        index: InMemoryVectorIndex | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.index = index or InMemoryVectorIndex()

    def index_documents(self, documents: list[SourceDocument]) -> None:
        if not documents:
            return

        embeddings = self.embedding_provider.embed([document.text for document in documents])
        if len(embeddings) != len(documents):
            raise RuntimeError("embedding provider returned the wrong number of vectors")

        for document, embedding in zip(documents, embeddings):
            self.index.add(
                document.document_id,
                document.text,
                embedding,
                metadata=document.metadata,
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

        query_embeddings = self.embedding_provider.embed([query])
        if len(query_embeddings) != 1:
            raise RuntimeError("embedding provider must return exactly one query vector")

        return self.index.search(
            query_embeddings[0],
            k=k,
            metadata_filter=metadata_filter,
        )
