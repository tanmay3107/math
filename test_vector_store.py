import json
from typing import Dict, List, Any, Optional
from metrics import VectorMetrics

class VectorStore:
    """Lightweight in-memory vector database with JSON persistence and similarity search."""

    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds or updates a vector with optional metadata in the store."""
        self._vectors[vector_id] = vector
        if metadata is not None:
            self._metadata[vector_id] = metadata

    def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a vector and its associated metadata by ID."""
        if vector_id not in self._vectors:
            return None
        return {
            "id": vector_id,
            "vector": self._vectors[vector_id],
            "metadata": self._metadata.get(vector_id, {})
        }

    def search(self, query_vector: List[float], k: int = 3, metric: str = "cosine") -> List[Dict[str, Any]]:
        """
        Searches for the top-k nearest vectors using the specified metric.
        Supported metrics: 'cosine', 'euclidean', 'manhattan'.
        """
        if not self._vectors:
            return []

        results = []
        for vec_id, vec in self._vectors.items():
            if metric == "cosine":
                score = VectorMetrics.cosine_similarity(query_vector, vec)
            elif metric == "euclidean":
                score = VectorMetrics.euclidean_distance(query_vector, vec)
            elif metric == "manhattan":
                score = VectorMetrics.manhattan_distance(query_vector, vec)
            else:
                raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine', 'euclidean', or 'manhattan'.")

            results.append({
                "id": vec_id,
                "score": score,
                "metadata": self._metadata.get(vec_id, {})
            })

        reverse_sort = (metric == "cosine")
        results.sort(key=lambda item: item["score"], reverse=reverse_sort)

        return results[:k]

    def save_to_json(self, filepath: str) -> None:
        """Saves the current state of vectors and metadata to a JSON file."""
        data = {
            "vectors": self._vectors,
            "metadata": self._metadata
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        """Loads state into the current instance from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._vectors = data.get("vectors", {})
        self._metadata = data.get("metadata", {})

    @classmethod
    def from_json(cls, filepath: str) -> "VectorStore":
        """Factory method to construct a new VectorStore instance from a JSON file."""
        store = cls()
        store.load_from_json(filepath)
        return store


if __name__ == "__main__":
    # Test saving and loading
    store = VectorStore()
    store.add("doc_1", [0.9, 0.1, 0.0], {"title": "Introduction to AI"})
    store.add("doc_2", [0.0, 0.1, 0.95], {"title": "Baking Sourdough Bread"})

    # Save to disk
    store.save_to_json("vector_store_backup.json")
    print("Saved store to vector_store_backup.json")

    # Load from disk into a fresh instance
    new_store = VectorStore.from_json("vector_store_backup.json")
    print("Loaded document from file:", new_store.get("doc_1"))