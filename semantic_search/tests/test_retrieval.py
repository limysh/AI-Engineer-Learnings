import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import RetrievalCase, evaluate_retrieval
from retrieval import InMemoryVectorIndex, cosine_similarity


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = InMemoryVectorIndex()
        self.index.add(
            "mountain",
            "Mountain hiking",
            [1.0, 0.0, 0.0],
            metadata={"country": "Canada"},
        )
        self.index.add(
            "transit",
            "City transit",
            [0.0, 1.0, 0.0],
            metadata={"country": "Japan"},
        )
        self.index.add(
            "food",
            "Cafe culture",
            [0.0, 0.0, 1.0],
            metadata={"country": "Canada"},
        )

    def test_cosine_similarity(self):
        self.assertAlmostEqual(1.0, cosine_similarity([1, 0], [2, 0]))
        self.assertAlmostEqual(0.0, cosine_similarity([1, 0], [0, 1]))

    def test_search_orders_by_similarity(self):
        results = self.index.search([0.9, 0.1, 0.0], k=2)

        self.assertEqual(["mountain", "transit"], [r.document_id for r in results])
        self.assertGreater(results[0].score, results[1].score)

    def test_metadata_filter_limits_candidates(self):
        results = self.index.search(
            [0.0, 1.0, 0.0],
            k=3,
            metadata_filter={"country": "Canada"},
        )

        self.assertEqual({"mountain", "food"}, {r.document_id for r in results})
        self.assertNotIn("transit", [r.document_id for r in results])

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.search([1.0, 0.0], k=1)

    def test_duplicate_document_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.index.add("mountain", "duplicate", [1.0, 0.0, 0.0])

    def test_evaluation_calculates_hit_rate_and_mrr(self):
        cases = [
            RetrievalCase(
                name="mountain query",
                query_embedding=[1.0, 0.0, 0.0],
                relevant_document_ids=frozenset({"mountain"}),
            ),
            RetrievalCase(
                name="food query",
                query_embedding=[0.0, 0.0, 1.0],
                relevant_document_ids=frozenset({"food"}),
            ),
        ]

        metrics = evaluate_retrieval(self.index, cases, k=2)

        self.assertEqual(2, metrics.cases)
        self.assertEqual(1.0, metrics.hit_rate_at_k)
        self.assertEqual(1.0, metrics.mean_reciprocal_rank)


if __name__ == "__main__":
    unittest.main()
