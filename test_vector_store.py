import unittest
import os
import tempfile
from vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    """Unit tests for the VectorStore class, including persistence methods."""

    def setUp(self):
        self.store = VectorStore()
        self.vec1 = [1.0, 0.0, 0.0]
        self.vec2 = [0.0, 1.0, 0.0]
        self.vec3 = [0.7071, 0.7071, 0.0]
        # Create a temporary directory for test file I/O
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up temporary directory after tests complete
        self.temp_dir.cleanup()

    def test_add_and_get(self):
        self.store.add("item_1", self.vec1, {"category": "A"})
        retrieved = self.store.get("item_1")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["id"], "item_1")
        self.assertEqual(retrieved["vector"], self.vec1)
        self.assertEqual(retrieved["metadata"], {"category": "A"})

    def test_get_nonexistent(self):
        self.assertIsNone(self.store.get("missing_id"))

    def test_search_cosine_ordering(self):
        self.store.add("vec1", self.vec1)
        self.store.add("vec2", self.vec2)
        self.store.add("vec3", self.vec3)

        results = self.store.search([0.9, 0.1, 0.0], k=2, metric="cosine")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "vec1")
        self.assertEqual(results[1]["id"], "vec3")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_search_distance_ordering(self):
        self.store.add("vec1", self.vec1)
        self.store.add("vec2", self.vec2)

        results_euc = self.store.search([0.9, 0.0, 0.0], k=2, metric="euclidean")
        self.assertEqual(results_euc[0]["id"], "vec1")
        self.assertLess(results_euc[0]["score"], results_euc[1]["score"])

        results_man = self.store.search([0.9, 0.0, 0.0], k=2, metric="manhattan")
        self.assertEqual(results_man[0]["id"], "vec1")
        self.assertLess(results_man[0]["score"], results_man[1]["score"])

    def test_empty_search(self):
        results = self.store.search([1.0, 0.0, 0.0], k=3)
        self.assertEqual(results, [])

    def test_invalid_metric_raises_error(self):
        self.store.add("vec1", self.vec1)
        with self.assertRaises(ValueError):
            self.store.search(self.vec1, metric="invalid_metric")

    def test_save_and_load_json(self):
        self.store.add("vec1", self.vec1, {"category": "math"})
        self.store.add("vec2", self.vec2, {"category": "physics"})

        filepath = os.path.join(self.temp_dir.name, "test_store.json")
        self.store.save_to_json(filepath)

        # Verify loading into a clean instance restores state accurately
        new_store = VectorStore()
        new_store.load_from_json(filepath)

        self.assertEqual(new_store.get("vec1"), self.store.get("vec1"))
        self.assertEqual(new_store.get("vec2"), self.store.get("vec2"))

    def test_from_json_factory(self):
        self.store.add("vec1", self.vec1, {"category": "math"})
        filepath = os.path.join(self.temp_dir.name, "test_store_factory.json")
        self.store.save_to_json(filepath)

        # Verify class factory method constructs and populates store
        loaded_store = VectorStore.from_json(filepath)
        self.assertEqual(loaded_store.get("vec1"), self.store.get("vec1"))

if __name__ == "__main__":
    unittest.main()