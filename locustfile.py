import random
from locust import HttpUser, task, between

class VectorStoreUser(HttpUser):
    wait_time = between(0.01, 0.05)  # Simulate high throughput

    def _random_vector(self, dim: int = 16):
        return [round(random.uniform(-1.0, 1.0), 4) for _ in range(dim)]

    @task(5)
    def search_hnsw(self):
        self.client.post("/search/dense", json={
            "query_vector": self._random_vector(),
            "k": 5,
            "backend": "hnsw"
        })

    @task(3)
    def search_hybrid(self):
        self.client.post("/search/hybrid", json={
            "query_vector": self._random_vector(),
            "query_text": "sample tech",
            "k": 5,
            "backend": "hnsw"
        })

    @task(2)
    def search_keyword(self):
        self.client.post("/search/keyword", json={
            "query_text": "document",
            "k": 5
        })

    @task(1)
    def insert_vector(self):
        doc_id = f"locust_{random.randint(10000, 99999)}"
        self.client.post("/vectors", json={
            "id": doc_id,
            "vector": self._random_vector(),
            "metadata": {"category": "load_test"}
        })