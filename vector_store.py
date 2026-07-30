from typing import Dict, List, Any, Optional
from metrics import VectorMetrics

class VectorStore:
    """Lightweight in-memory vector database for similarity search."""

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

        # Higher similarity is better for cosine; lower distance is better for Euclidean/Manhattan
        reverse_sort = (metric == "cosine")
        results.sort(key=lambda item: item["score"], reverse=reverse_sort)

        return results[:k]


if __name__ == "__main__":
    store = VectorStore()
    
    # Store sample embedding vectors representing topics
    store.add("doc_1", [0.9, 0.1, 0.0], {"title": "Introduction to AI"})
    store.add("doc_2", [0.85, 0.15, 0.05], {"title": "Machine Learning Fundamentals"})
    store.add("doc_3", [0.0, 0.1, 0.95], {"title": "Baking Sourdough Bread"})

    query = [0.92, 0.08, 0.0]
    top_results = store.search(query_vector=query, k=2, metric="cosine")

    print(f"Top results for query {query}:")
    for res in top_results:
        print(f" - [{res['id']}] Score: {res['score']:.4f} | Title: {res['metadata'].get('title')}")