import unittest
import os
import tempfile
from vector_store import VectorStore

class TestVectorStoreHNSW(unittest.TestCase):
    """Unit tests for VectorStore integration with the HNSW ANN search backend."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = VectorStore(use_hnsw=True, hnsw_kwargs={"distance_metric": "cosine", "M": 16})
        
        # Seed test entries with vectors and metadata
        self.store.add("doc1", [1.0, 0.0, 0.0], {"category": "tech", "lang": "en"})
        self.store.add("doc2", [0.9, 0.1, 0.0], {"category": "tech", "lang": "es"})
        self.store.add("doc3", [0.0, 1.0, 0.0], {"category": "finance", "lang": "en"})
        self.store.add("doc4", [0.0, 0.0, 1.0], {"category": "finance", "lang": "es"})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hnsw_backend_search(self):
        results = self.store.search([0.95, 0.05, 0.0], k=2, backend="hnsw")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc1")
        self.assertEqual(results[1]["id"], "doc2")

    def test_hnsw_backend_not_enabled_raises(self):
        exact_store = VectorStore(use_hnsw=False)
        exact_store.add("doc1", [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            exact_store.search([1.0, 0.0, 0.0], backend="hnsw")

    def test_hnsw_search_with_metadata_filter(self):
        # Closest to doc1 [1,0,0], but filter requires lang=es
        results = self.store.search(
            query_vector=[1.0, 0.0, 0.0],
            k=2,
            filter_metadata={"lang": "es"},
            backend="hnsw"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc2")
        self.assertEqual(results[0]["metadata"]["lang"], "es")

    def test_hnsw_hybrid_search(self):
        results = self.store.hybrid_search(
            query_vector=[0.0, 1.0, 0.0],
            query_text="finance",
            k=2,
            backend="hnsw"
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "doc3")

    def test_hnsw_delete_rebuilds_graph(self):
        self.store.delete("doc1")
        results = self.store.search([1.0, 0.0, 0.0], k=1, backend="hnsw")
        self.assertEqual(results[0]["id"], "doc2")

    def test_hnsw_update_rebuilds_graph(self):
        # Redirect doc1 vector toward finance
        self.store.update("doc1", vector=[0.0, 0.99, 0.01])
        results = self.store.search([0.0, 1.0, 0.0], k=1, backend="hnsw")
        self.assertEqual(results[0]["id"], "doc1")

    def test_hnsw_persistence_reload(self):
        filepath = os.path.join(self.temp_dir.name, "store_hnsw.json")
        self.store.save_to_json(filepath)

        # Reload store state with HNSW enabled
        reloaded_store = VectorStore.from_json(filepath, use_hnsw=True)
        results = reloaded_store.search([1.0, 0.0, 0.0], k=1, backend="hnsw")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1")

if __name__ == "__main__":
    unittest.main()