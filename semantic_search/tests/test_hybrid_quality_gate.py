import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import RetrievalQualityError
from hybrid_quality_gate import DEFAULT_DATASET, run_hybrid_quality_gate


class HybridQualityGateTests(unittest.TestCase):
    def _run_modified(self, modify):
        dataset = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        modify(dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dataset.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            return run_hybrid_quality_gate(path)

    def test_checked_in_cases_prove_lexical_rescue_without_semantic_regression(self):
        report = run_hybrid_quality_gate()
        self.assertEqual(report.hybrid.cases, 4)
        self.assertEqual(report.vector.hit_rate_at_k, 0.75)
        self.assertEqual(report.hybrid.hit_rate_at_k, 1.0)
        self.assertEqual(report.hybrid.mean_reciprocal_rank, 1.0)
        self.assertEqual(report.rescued_cases, ("exact incident code rescued by BM25",))

    def test_quality_floor_catches_loss_of_exact_identifier(self):
        def mutate(dataset):
            dataset["cases"][0]["query_text"] = "unrelated"

        with self.assertRaisesRegex(RetrievalQualityError, "Hit Rate@K"):
            self._run_modified(mutate)

    def test_rescue_count_is_an_explicit_gate(self):
        def mutate(dataset):
            dataset["thresholds"]["min_rescued_cases"] = 2

        with self.assertRaisesRegex(RetrievalQualityError, "rescued 1 cases"):
            self._run_modified(mutate)

    def test_empty_golden_set_fails_instead_of_dividing_by_zero(self):
        def mutate(dataset):
            dataset["cases"] = []

        with self.assertRaisesRegex(ValueError, "at least one case"):
            self._run_modified(mutate)


if __name__ == "__main__":
    unittest.main()
