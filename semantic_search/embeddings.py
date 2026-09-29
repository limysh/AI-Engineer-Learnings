"""Embedding provider boundaries for semantic retrieval.

The retrieval core should not depend directly on one model vendor. This module
keeps external embedding calls behind a small interface so ranking and evaluation
remain deterministic and easy to test.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


Embedding = tuple[float, ...]


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Return one embedding per input text, preserving input order."""


class OpenAIEmbeddingProvider:
    """Thin adapter around an OpenAI-compatible embeddings client."""

    def __init__(self, client: Any, *, model: str = "text-embedding-3-small") -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.client = client
        self.model = model

    @classmethod
    def from_environment(
        cls,
        *,
        model: str = "text-embedding-3-small",
    ) -> "OpenAIEmbeddingProvider":
        """Build the adapter using the SDK's normal environment configuration."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for live embedding calls"
            ) from exc

        return cls(OpenAI(), model=model)

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        inputs = list(texts)
        if not inputs:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in inputs):
            raise ValueError("embedding inputs must be non-empty strings")

        response = self.client.embeddings.create(
            model=self.model,
            input=inputs,
        )
        items = list(response.data)

        if len(items) != len(inputs):
            raise RuntimeError(
                f"embedding provider returned {len(items)} vectors for {len(inputs)} inputs"
            )

        by_index: dict[int, Embedding] = {}
        for fallback_index, item in enumerate(items):
            index = getattr(item, "index", fallback_index)
            if not isinstance(index, int) or index < 0 or index >= len(inputs):
                raise RuntimeError(f"embedding provider returned invalid index: {index!r}")
            if index in by_index:
                raise RuntimeError(f"embedding provider returned duplicate index: {index}")

            vector = tuple(float(value) for value in item.embedding)
            if not vector:
                raise RuntimeError("embedding provider returned an empty vector")
            by_index[index] = vector

        vectors = [by_index[index] for index in range(len(inputs))]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError("embedding provider returned inconsistent vector dimensions")

        return vectors
