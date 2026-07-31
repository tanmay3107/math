import unittest
from vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    """Unit tests for the VectorStore class."""

    def setUp(self):
        self.store = VectorStore()
        self.vec1 = [1.0, 0.0, 0.0]
        self.vec2 = [0.0, 1.0, 0.0]
        self.vec3 = [0.7071, 0.7071, 0.0]

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

        # Query vector close to vec1
        results = self.store.search([0.9, 0.1, 0.0], k=2, metric="cosine")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "vec1")
        self.assertEqual(results[1]["id"], "vec3")
        # Cosine similarity ranks highest score first
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_search_distance_ordering(self):
        self.store.add("vec1", self.vec1)
        self.store.add("vec2", self.vec2)

        # Distance metrics should rank lowest score (closest distance) first
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

if __name__ == "__main__":
    unittest.main()