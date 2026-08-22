import unittest
from vector_store import VectorStore

class TestHybridSearch(unittest.TestCase):
    """Unit tests for BM25 inverted index keyword search and RRF hybrid search in VectorStore."""

    def setUp(self):
        self.store = VectorStore()
        # Add sample technical documents with distinct vector representations and metadata
        self.store.add(
            "doc_ai",
            [1.0, 0.0, 0.0],
            {"title": "Artificial Intelligence Basics", "content": "Introduction to AI and machine learning models."}
        )
        self.store.add(
            "doc_ml",
            [0.8, 0.6, 0.0],
            {"title": "Machine Learning Pipelines", "content": "Building scalable ML training pipelines and workflows."}
        )
        self.store.add(
            "doc_db",
            [0.0, 0.0, 1.0],
            {"title": "Vector Databases", "content": "Indexing high-dimensional embeddings for similarity search."}
        )

    def test_keyword_search_basic(self):
        results = self.store.keyword_search("machine learning", k=2)
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("doc_ml", result_ids)
        self.assertIn("doc_ai", result_ids)

    def test_keyword_search_no_match(self):
        results = self.store.keyword_search("quantum computing", k=2)
        self.assertEqual(results, [])

    def test_keyword_search_case_and_punctuation_insensitive(self):
        results = self.store.keyword_search("ARTIFICIAL!! intelligence...", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_ai")

    def test_inverted_index_update_on_delete(self):
        # Confirm match exists before deletion
        results_before = self.store.keyword_search("embeddings", k=5)
        self.assertEqual(len(results_before), 1)

        # Delete document and verify terms are purged from the inverted index
        self.store.delete("doc_db")
        results_after = self.store.keyword_search("embeddings", k=5)
        self.assertEqual(results_after, [])

    def test_inverted_index_update_on_modify(self):
        # Replace metadata for doc_ai
        self.store.update("doc_ai", metadata={"title": "Deep Learning Frameworks", "content": "Neural networks and PyTorch."})

        # Old keywords should no longer produce matches
        old_match = self.store.keyword_search("Artificial", k=1)
        self.assertEqual(len(old_match), 0)

        # New keywords should be indexed and searchable
        new_match = self.store.keyword_search("PyTorch", k=1)
        self.assertEqual(len(new_match), 1)
        self.assertEqual(new_match[0]["id"], "doc_ai")

    def test_hybrid_search_rrf_combination(self):
        # Query vector is closest to doc_db ([0.0, 0.0, 1.0]), but query text matches doc_ml ("pipelines")
        results = self.store.hybrid_search(
            query_vector=[0.0, 0.0, 1.0],
            query_text="pipelines",
            k=3
        )
        
        self.assertGreater(len(results), 0)
        
        # Verify RRF scores were computed and stored
        for res in results:
            self.assertIn("rrf_score", res)
            self.assertGreater(res["rrf_score"], 0.0)

        result_ids = [r["id"] for r in results]
        # Both top vector match (doc_db) and top keyword match (doc_ml) should appear in hybrid results
        self.assertIn("doc_db", result_ids)
        self.assertIn("doc_ml", result_ids)

    def test_hybrid_search_empty_store(self):
        empty_store = VectorStore()
        results = empty_store.hybrid_search([1.0, 0.0, 0.0], "test query", k=3)
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()