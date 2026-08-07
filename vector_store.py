import json
from typing import Dict, List, Any, Optional
from metrics import VectorMetrics

class VectorStore:
    """Lightweight in-memory vector database with persistence, metadata filtering, and search."""

    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_vector(vector: List[float]) -> List[float]:
        """Helper to convert a vector into a unit vector (L2 normalization)."""
        mag = VectorMetrics.magnitude(vector)
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return [v / mag for v in vector]

    def _matches_filter(self, vec_id: str, filter_metadata: Dict[str, Any]) -> bool:
        """Helper to check if a vector's metadata matches all criteria in filter_metadata."""
        item_meta = self._metadata.get(vec_id, {})
        return all(item_meta.get(key) == value for key, value in filter_metadata.items())

    def add(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Optional[Dict[str, Any]] = None,
        normalize: bool = False
    ) -> None:
        """Adds or updates a vector with optional metadata and optional L2 normalization."""
        if normalize:
            vector = self._normalize_vector(vector)
            
        self._vectors[vector_id] = vector
        if metadata is not None:
            self._metadata[vector_id] = metadata

    def add_batch(
        self, 
        records: List[Dict[str, Any]], 
        normalize: bool = False
    ) -> None:
        """Batch adds multiple vector records to the store."""
        for record in records:
            if "id" not in record or "vector" not in record:
                raise KeyError("Each record in batch must contain 'id' and 'vector' keys.")
            self.add(
                vector_id=record["id"],
                vector=record["vector"],
                metadata=record.get("metadata"),
                normalize=normalize
            )

    def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a vector and its associated metadata by ID."""
        if vector_id not in self._vectors:
            return None
        return {
            "id": vector_id,
            "vector": self._vectors[vector_id],
            "metadata": self._metadata.get(vector_id, {})
        }

    def search(
        self, 
        query_vector: List[float], 
        k: int = 3, 
        metric: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for top-k nearest vectors using the specified metric and optional metadata filters.
        Supported metrics: 'cosine', 'euclidean', 'manhattan'.
        """
        if not self._vectors:
            return []

        results = []
        for vec_id, vec in self._vectors.items():
            # Apply metadata filtering before calculating distance/similarity
            if filter_metadata and not self._matches_filter(vec_id, filter_metadata):
                continue

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
    store = VectorStore()

    store.add("doc_1", [1.0, 0.0, 0.0], {"category": "tech", "author": "alice"})
    store.add("doc_2", [0.99, 0.01, 0.0], {"category": "cooking", "author": "bob"})
    store.add("doc_3", [0.95, 0.05, 0.0], {"category": "tech", "author": "alice"})

    query = [1.0, 0.0, 0.0]

    # Search filtered by category="tech"
    filtered_results = store.search(query, k=2, filter_metadata={"category": "tech"})

    print("Filtered Search Results (tech category only):")
    for res in filtered_results:
        print(f" - [{res['id']}] Score: {res['score']:.4f} | Meta: {res['metadata']}")