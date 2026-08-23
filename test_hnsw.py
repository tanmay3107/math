import unittest
import random
from hnsw import HNSWIndex
from metrics import VectorMetrics

class TestHNSWIndex(unittest.TestCase):
    """Unit tests and Recall@k benchmarks comparing HNSW ANN search against exact KNN search."""

    def setUp(self):
        random.seed(42)
        self.dim = 16
        self.index = HNSWIndex(distance_metric="cosine", M=16, ef_construction=64, ef_search=32)

    def _random_vector(self, dim=16):
        return [random.uniform(-1.0, 1.0) for _ in range(dim)]

    def test_basic_add_and_search(self):
        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]
        index = HNSWIndex(distance_metric="cosine")
        index.add("v1", vec1)
        index.add("v2", vec2)

        results = index.search([0.9, 0.1, 0.0, 0.0], k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "v1")

    def test_empty_search(self):
        index = HNSWIndex()
        results = index.search([1.0, 0.0, 0.0], k=3)
        self.assertEqual(results, [])

    def test_supported_metrics(self):
        for metric in ["cosine", "euclidean", "manhattan"]:
            idx = HNSWIndex(distance_metric=metric)
            idx.add("a", [1.0, 0.0])
            idx.add("b", [0.0, 1.0])
            res = idx.search([0.9, 0.1], k=1)
            self.assertEqual(res[0]["id"], "a")

    def test_hnsw_recall_benchmark(self):
        num_vectors = 200
        k = 5
        vectors = {}

        # 1. Populate HNSW graph index and keep in-memory vector reference
        for i in range(num_vectors):
            vec_id = f"vec_{i}"
            vec = self._random_vector(self.dim)
            vectors[vec_id] = vec
            self.index.add(vec_id, vec)

        queries = [self._random_vector(self.dim) for _ in range(20)]
        total_recall = 0.0

        for query in queries:
            # 2. Compute Ground Truth via brute-force exact search
            exact_distances = []
            for vec_id, vec in vectors.items():
                dist = 1.0 - VectorMetrics.cosine_similarity(query, vec)
                exact_distances.append((dist, vec_id))
            exact_distances.sort(key=lambda x: x[0])
            ground_truth_ids = set([vid for _, vid in exact_distances[:k]])

            # 3. Execute HNSW approximate search
            hnsw_results = self.index.search(query, k=k)
            hnsw_ids = set([r["id"] for r in hnsw_results])

            # 4. Calculate Recall@k ratio
            overlap = len(ground_truth_ids.intersection(hnsw_ids))
            total_recall += overlap / k

        average_recall = total_recall / len(queries)
        print(f"\n[HNSW Benchmark] Average Recall@{k} across {len(queries)} queries: {average_recall * 100:.2f}%")

        # HNSW should achieve >= 85% recall on small datasets
        self.assertGreaterEqual(average_recall, 0.85)

if __name__ == "__main__":
    unittest.main()