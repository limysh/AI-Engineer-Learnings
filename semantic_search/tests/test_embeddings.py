import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings import OpenAIEmbeddingProvider


class FakeEmbeddingsApi:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": list(input)})
        return SimpleNamespace(data=self.data)


class OpenAIEmbeddingProviderTests(unittest.TestCase):
    def test_batch_embeddings_preserve_input_order(self):
        api = FakeEmbeddingsApi(
            [
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )
        client = SimpleNamespace(embeddings=api)
        provider = OpenAIEmbeddingProvider(client, model="embedding-model")

        vectors = provider.embed(["first", "second"])

        self.assertEqual([(1.0, 0.0), (0.0, 1.0)], vectors)
        self.assertEqual(
            [{"model": "embedding-model", "input": ["first", "second"]}],
            api.calls,
        )

    def test_missing_embedding_is_rejected(self):
        api = FakeEmbeddingsApi(
            [SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        )
        provider = OpenAIEmbeddingProvider(SimpleNamespace(embeddings=api))

        with self.assertRaises(RuntimeError):
            provider.embed(["first", "second"])

    def test_inconsistent_dimensions_are_rejected(self):
        api = FakeEmbeddingsApi(
            [
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
            ]
        )
        provider = OpenAIEmbeddingProvider(SimpleNamespace(embeddings=api))

        with self.assertRaises(RuntimeError):
            provider.embed(["first", "second"])

    def test_blank_embedding_input_is_rejected(self):
        api = FakeEmbeddingsApi([])
        provider = OpenAIEmbeddingProvider(SimpleNamespace(embeddings=api))

        with self.assertRaises(ValueError):
            provider.embed(["   "])


if __name__ == "__main__":
    unittest.main()
