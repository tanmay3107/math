import unittest
import os
import tempfile
from fastapi.testclient import TestClient
from app import app, store

class TestFastAPIApp(unittest.TestCase):
    """Automated unit tests for FastAPI REST API endpoints using TestClient."""

    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Reset in-memory store state before each test
        store.vectors.clear()
        store.metadata.clear()
        store.inverted_index.clear()
        store.doc_term_freqs.clear()
        store.doc_lengths.clear()
        if store.use_hnsw:
            store._rebuild_hnsw()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["total_documents"], 0)
        self.assertTrue(data["hnsw_enabled"])

    def test_add_and_get_vector(self):
        payload = {
            "id": "doc1",
            "vector": [1.0, 0.0, 0.0],
            "metadata": {"topic": "python", "author": "dev"}
        }
        add_res = self.client.post("/vectors", json=payload)
        self.assertEqual(add_res.status_code, 201)
        self.assertEqual(add_res.json()["id"], "doc1")

        get_res = self.client.get("/vectors/doc1")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()
        self.assertEqual(data["id"], "doc1")
        self.assertEqual(data["vector"], [1.0, 0.0, 0.0])
        self.assertEqual(data["metadata"]["topic"], "python")

    def test_get_vector_not_found(self):
        response = self.client.get("/vectors/non_existent")
        self.assertEqual(response.status_code, 404)

    def test_add_batch(self):
        payload = {
            "records": [
                {"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"tag": "a"}},
                {"id": "doc2", "vector": [0.0, 1.0, 0.0], "metadata": {"tag": "b"}}
            ],
            "normalize": False
        }
        response = self.client.post("/vectors/batch", json=payload)
        self.assertEqual(response.status_code, 201)

        root_res = self.client.get("/")
        self.assertEqual(root_res.json()["total_documents"], 2)

    def test_update_vector(self):
        self.client.post("/vectors", json={"id": "doc1", "vector": [1.0, 0.0, 0.0]})

        update_payload = {"vector": [0.0, 1.0, 0.0], "metadata": {"status": "updated"}}
        put_res = self.client.put("/vectors/doc1", json=update_payload)
        self.assertEqual(put_res.status_code, 200)

        get_res = self.client.get("/vectors/doc1")
        self.assertEqual(get_res.json()["vector"], [0.0, 1.0, 0.0])
        self.assertEqual(get_res.json()["metadata"]["status"], "updated")

    def test_delete_vector(self):
        self.client.post("/vectors", json={"id": "doc1", "vector": [1.0, 0.0, 0.0]})

        del_res = self.client.delete("/vectors/doc1")
        self.assertEqual(del_res.status_code, 200)

        get_res = self.client.get("/vectors/doc1")
        self.assertEqual(get_res.status_code, 404)

    def test_search_dense_exact_and_hnsw(self):
        self.client.post("/vectors", json={"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"category": "tech"}})
        self.client.post("/vectors", json={"id": "doc2", "vector": [0.9, 0.1, 0.0], "metadata": {"category": "tech"}})
        self.client.post("/vectors", json={"id": "doc3", "vector": [0.0, 1.0, 0.0], "metadata": {"category": "finance"}})

        # Test exact search
        exact_req = {"query_vector": [1.0, 0.0, 0.0], "k": 2, "backend": "exact"}
        exact_res = self.client.post("/search/dense", json=exact_req)
        self.assertEqual(exact_res.status_code, 200)
        self.assertEqual(len(exact_res.json()["results"]), 2)
        self.assertEqual(exact_res.json()["results"][0]["id"], "doc1")

        # Test HNSW search with metadata filter
        hnsw_req = {
            "query_vector": [1.0, 0.0, 0.0],
            "k": 2,
            "filter_metadata": {"category": "tech"},
            "backend": "hnsw"
        }
        hnsw_res = self.client.post("/search/dense", json=hnsw_req)
        self.assertEqual(hnsw_res.status_code, 200)
        results = hnsw_res.json()["results"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["metadata"]["category"] == "tech" for r in results))

    def test_search_keyword(self):
        self.client.post("/vectors", json={"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"text": "machine learning"}})
        self.client.post("/vectors", json={"id": "doc2", "vector": [0.0, 1.0, 0.0], "metadata": {"text": "deep learning"}})

        req = {"query_text": "machine", "k": 1}
        res = self.client.post("/search/keyword", json=req)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["results"]), 1)
        self.assertEqual(res.json()["results"][0]["id"], "doc1")

    def test_search_hybrid(self):
        self.client.post("/vectors", json={"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"text": "vector search engine"}})

        req = {
            "query_vector": [1.0, 0.0, 0.0],
            "query_text": "vector search",
            "k": 1,
            "backend": "hnsw"
        }
        res = self.client.post("/search/hybrid", json=req)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["results"][0]["id"], "doc1")

    def test_save_and_load_store(self):
        filepath = os.path.join(self.temp_dir.name, "api_store.json")
        self.client.post("/vectors", json={"id": "doc1", "vector": [1.0, 0.0, 0.0]})

        # Save store via endpoint
        save_res = self.client.post("/store/save", json={"filepath": filepath})
        self.assertEqual(save_res.status_code, 200)

        # Clear store state manually
        store.vectors.clear()

        # Load store state via endpoint
        load_res = self.client.post("/store/load", json={"filepath": filepath})
        self.assertEqual(load_res.status_code, 200)

        # Confirm data restored
        get_res = self.client.get("/vectors/doc1")
        self.assertEqual(get_res.status_code, 200)

if __name__ == "__main__":
    unittest.main()