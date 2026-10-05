import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid import HybridSearchIndex, tokenize


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.index = HybridSearchIndex()
        self.index.add(
            "finance",
            "Quarterly revenue report",
            [1.0, 0.0, 0.0],
            metadata={"team": "finance"},
        )
        self.index.add(
            "vacation",
            "Employee vacation policy",
            [0.9, 0.1, 0.0],
            metadata={"team": "people"},
        )
        self.index.add(
            "mountain",
            "Rocky Mountain hiking guide",
            [0.0, 0.0, 1.0],
            metadata={"team": "travel"},
        )

    def test_exact_terms_rescue_relevant_result_from_vector_miss(self):
        results = self.index.search(
            "vacation policy",
            [1.0, 0.0, 0.0],
            k=2,
        )

        self.assertEqual("vacation", results[0].document_id)
        self.assertEqual(2, results[0].vector_rank)
        self.assertEqual(1, results[0].lexical_rank)

    def test_vector_ranking_handles_semantic_query_without_term_overlap(self):
        results = self.index.search("alpine trails", [0.0, 0.1, 0.9], k=1)

        self.assertEqual("mountain", results[0].document_id)
        self.assertEqual(1, results[0].vector_rank)
        self.assertIsNone(results[0].lexical_rank)

    def test_metadata_filter_applies_to_both_rankings(self):
        results = self.index.search(
            "vacation policy",
            [1.0, 0.0, 0.0],
            k=3,
            metadata_filter={"team": "finance"},
        )

        self.assertEqual(["finance"], [result.document_id for result in results])

    def test_tokenization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(
            ["employee", "vacation", "policy", "2026"],
            tokenize("Employee VACATION-policy, 2026!"),
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridSearchIndex(rrf_k=0)
        with self.assertRaises(ValueError):
            HybridSearchIndex(bm25_b=1.1)


if __name__ == "__main__":
    unittest.main()
