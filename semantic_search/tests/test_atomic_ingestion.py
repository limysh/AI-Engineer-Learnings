import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval import Document, InMemoryVectorIndex, cosine_similarity
from service import SemanticSearchService, SourceDocument


class AtomicIngestionTests(unittest.TestCase):
    def setUp(self):
        self.index = InMemoryVectorIndex()
        self.index.add("existing", "Existing document", [1.0, 0.0])

    def test_valid_batch_commits_all_documents(self):
        self.index.add_many([
            Document("one", "One", (0.0, 1.0)),
            Document("two", "Two", (0.5, 0.5)),
        ])
        self.assertEqual(
            {"existing", "one", "two"},
            {result.document_id for result in self.index.search([1.0, 0.0], k=3)},
        )

    def test_dimension_error_in_later_document_rolls_back_entire_batch(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.index.add_many([
                Document("one", "One", (0.0, 1.0)),
                Document("bad", "Bad", (1.0, 0.0, 0.0)),
            ])
        self.assertEqual(["existing"], [r.document_id for r in self.index.search([1.0, 0.0])])
        self.index.add("one", "Can retry after failed batch", [0.0, 1.0])

    def test_duplicate_in_batch_rolls_back_entire_batch(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.index.add_many([
                Document("new", "New", (0.0, 1.0)),
                Document("new", "Duplicate", (0.0, 1.0)),
            ])
        self.assertEqual(["existing"], [r.document_id for r in self.index.search([1.0, 0.0])])

    def test_duplicate_of_existing_document_rolls_back_new_documents(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.index.add_many([
                Document("new", "New", (0.0, 1.0)),
                Document("existing", "Duplicate", (1.0, 0.0)),
            ])
        self.assertEqual(["existing"], [r.document_id for r in self.index.search([1.0, 0.0])])

    def test_nonfinite_vectors_are_rejected_without_mutation(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.index.add_many([
                        Document("new", "New", (0.0, 1.0)),
                        Document("bad", "Bad", (value, 0.0)),
                    ])
                self.assertEqual(["existing"], [r.document_id for r in self.index.search([1.0, 0.0])])

    def test_nonfinite_query_and_similarity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.index.search([math.nan, 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            cosine_similarity([1.0, 0.0], [math.inf, 0.0])

    def test_failed_first_batch_does_not_lock_index_dimension(self):
        fresh = InMemoryVectorIndex()
        with self.assertRaisesRegex(ValueError, "dimension"):
            fresh.add_many([
                Document("a", "A", (1.0, 0.0)),
                Document("b", "B", (0.0, 1.0, 0.0)),
            ])
        fresh.add("valid", "Valid", [1.0, 0.0, 0.0])
        self.assertEqual(["valid"], [r.document_id for r in fresh.search([1.0, 0.0, 0.0])])

    def test_service_does_not_partially_index_bad_embedding_batch(self):
        class Provider:
            def embed(self, texts):
                return [(1.0, 0.0), (0.0, 1.0, 0.0)]

        service = SemanticSearchService(Provider())
        with self.assertRaisesRegex(ValueError, "dimension"):
            service.index_documents([
                SourceDocument("first", "First"),
                SourceDocument("second", "Second"),
            ])
        self.assertEqual([], service.index.search([1.0, 0.0], k=2))


if __name__ == "__main__":
    unittest.main()
