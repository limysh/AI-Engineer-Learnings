import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import RetrievalQualityError, RetrievalThresholds
from quality_gate import DEFAULT_DATASET, run_quality_gate


class RetrievalQualityGateTests(unittest.TestCase):
    def test_checked_in_golden_dataset_passes(self):
        metrics = run_quality_gate(DEFAULT_DATASET)

        self.assertEqual(3, metrics.cases)
        self.assertEqual(1.0, metrics.hit_rate_at_k)
        self.assertEqual(1.0, metrics.mean_reciprocal_rank)

    def test_gate_reports_metric_regression(self):
        dataset = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        dataset["cases"][0]["relevant_document_ids"] = ["missing-document"]

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "regressed.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")

            with self.assertRaisesRegex(
                RetrievalQualityError,
                r"Hit Rate@K 0\.667 is below required 1\.000.*"
                r"Mean Reciprocal Rank 0\.667 is below required 1\.000",
            ):
                run_quality_gate(path)

    def test_thresholds_reject_invalid_range(self):
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            RetrievalThresholds(
                min_hit_rate_at_k=1.01,
                min_mean_reciprocal_rank=0.8,
            )


if __name__ == "__main__":
    unittest.main()
