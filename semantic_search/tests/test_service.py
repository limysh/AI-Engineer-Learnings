import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service import SemanticSearchService, SourceDocument


class FakeEmbeddingProvider:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        inputs = list(texts)
        self.calls.append(inputs)
        return [self.vectors[text] for text in inputs]


class SemanticSearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeEmbeddingProvider(
            {
                "Mountain hiking in Canada": (1.0, 0.0, 0.0),
                "Tokyo rail network": (0.0, 1.0, 0.0),
                "Montreal food and cafes": (0.0, 0.0, 1.0),
                "alpine trails": (0.95, 0.05, 0.0),
                "public transit": (0.05, 0.95, 0.0),
            }
        )
        self.service = SemanticSearchService(self.provider)
        self.service.index_documents(
            [
                SourceDocument(
                    "banff",
                    "Mountain hiking in Canada",
                    {"country": "Canada", "theme": "outdoors"},
                ),
                SourceDocument(
                    "tokyo",
                    "Tokyo rail network",
                    {"country": "Japan", "theme": "transit"},
                ),
                SourceDocument(
                    "montreal",
                    "Montreal food and cafes",
                    {"country": "Canada", "theme": "food"},
                ),
            ]
        )

    def test_text_query_is_embedded_and_ranked(self):
        results = self.service.search("alpine trails", k=2)

        self.assertEqual("banff", results[0].document_id)
        self.assertEqual(["alpine trails"], self.provider.calls[-1])

    def test_metadata_filter_applies_after_query_embedding(self):
        results = self.service.search(
            "public transit",
            k=3,
            metadata_filter={"country": "Canada"},
        )

        self.assertNotIn("tokyo", [result.document_id for result in results])
        self.assertEqual({"banff", "montreal"}, {result.document_id for result in results})

    def test_blank_query_is_rejected_before_provider_call(self):
        calls_before = len(self.provider.calls)

        with self.assertRaises(ValueError):
            self.service.search("   ")

        self.assertEqual(calls_before, len(self.provider.calls))


if __name__ == "__main__":
    unittest.main()
