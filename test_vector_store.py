import unittest
import os
import tempfile
import math
from vector_store import VectorStore
from metrics import VectorMetrics

class TestVectorStore(unittest.TestCase):
    """Unit tests for the VectorStore class, including persistence, normalization, batch operations, and filtering."""

    def setUp(self):
        self.store = VectorStore()
        self.vec1 = [1.0, 0.0, 0.0]
        self.vec2 = [0.0, 1.0, 0.0]
        self.vec3 = [0.7071, 0.7071, 0.0]
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
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

        new_store = VectorStore()
        new_store.load_from_json(filepath)

        self.assertEqual(new_store.get("vec1"), self.store.get("vec1"))
        self.assertEqual(new_store.get("vec2"), self.store.get("vec2"))

    def test_from_json_factory(self):
        self.store.add("vec1", self.vec1, {"category": "math"})
        filepath = os.path.join(self.temp_dir.name, "test_store_factory.json")
        self.store.save_to_json(filepath)

        loaded_store = VectorStore.from_json(filepath)
        self.assertEqual(loaded_store.get("vec1"), self.store.get("vec1"))

    def test_vector_normalization(self):
        self.store.add("norm_vec", [3.0, 4.0, 0.0], normalize=True)
        retrieved = self.store.get("norm_vec")
        
        self.assertIsNotNone(retrieved)
        self.assertAlmostEqual(VectorMetrics.magnitude(retrieved["vector"]), 1.0)
        self.assertAlmostEqual(retrieved["vector"][0], 0.6)
        self.assertAlmostEqual(retrieved["vector"][1], 0.8)

    def test_normalize_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            self.store.add("zero", [0.0, 0.0, 0.0], normalize=True)

    def test_add_batch(self):
        batch = [
            {"id": "b1", "vector": [3.0, 4.0, 0.0], "metadata": {"tag": "first"}},
            {"id": "b2", "vector": [0.0, 5.0, 0.0], "metadata": {"tag": "second"}}
        ]
        self.store.add_batch(batch, normalize=True)

        item1 = self.store.get("b1")
        item2 = self.store.get("b2")

        self.assertIsNotNone(item1)
        self.assertIsNotNone(item2)
        self.assertAlmostEqual(VectorMetrics.magnitude(item1["vector"]), 1.0)
        self.assertAlmostEqual(VectorMetrics.magnitude(item2["vector"]), 1.0)
        self.assertEqual(item1["metadata"]["tag"], "first")

    def test_add_batch_invalid_record_raises(self):
        invalid_batch = [
            {"vector": [1.0, 2.0, 3.0]}
        ]
        with self.assertRaises(KeyError):
            self.store.add_batch(invalid_batch)

    def test_search_with_single_metadata_filter(self):
        self.store.add("doc1", [1.0, 0.0, 0.0], {"category": "tech", "author": "alice"})
        self.store.add("doc2", [0.99, 0.01, 0.0], {"category": "cooking", "author": "bob"})
        self.store.add("doc3", [0.95, 0.05, 0.0], {"category": "tech", "author": "charlie"})

        results = self.store.search([1.0, 0.0, 0.0], k=3, filter_metadata={"category": "tech"})
        
        self.assertEqual(len(results), 2)
        result_ids = [r["id"] for r in results]
        self.assertIn("doc1", result_ids)
        self.assertIn("doc3", result_ids)
        self.assertNotIn("doc2", result_ids)

    def test_search_with_multiple_metadata_filters(self):
        self.store.add("doc1", [1.0, 0.0, 0.0], {"category": "tech", "author": "alice"})
        self.store.add("doc2", [0.99, 0.01, 0.0], {"category": "tech", "author": "bob"})

        results = self.store.search([1.0, 0.0, 0.0], k=3, filter_metadata={"category": "tech", "author": "alice"})
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1")

    def test_search_filter_no_matches(self):
        self.store.add("doc1", [1.0, 0.0, 0.0], {"category": "tech"})

        results = self.store.search([1.0, 0.0, 0.0], k=3, filter_metadata={"category": "sports"})
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()