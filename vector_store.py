from typing import List, Dict, Any, Optional
from metrics import VectorMetrics
from pq import ProductQuantizer

class VectorStore:
    """
    In-memory vector store supporting exact k-NN, BM25, HNSW, and PQ Quantized search.
    """

    def __init__(self, use_hnsw: bool = False, hnsw_kwargs: Optional[Dict] = None, use_pq: bool = False):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # HNSW configuration
        self.use_hnsw = use_hnsw
        self.hnsw_kwargs = hnsw_kwargs or {}
        self.hnsw_index = None

        # PQ configuration
        self.use_pq = use_pq
        self.pq: Optional[ProductQuantizer] = None
        self.pq_codes: Dict[str, List[int]] = {}

    def fit_pq(self, num_subvectors: int = 8, num_centroids: int = 256, max_iter: int = 20):
        """Fits the Product Quantizer on currently stored vectors and encodes the corpus."""
        if not self.vectors:
            raise ValueError("Cannot fit PQ on an empty store. Add vectors first.")

        vector_list = list(self.vectors.values())
        self.pq = ProductQuantizer(num_subvectors=num_subvectors, num_centroids=num_centroids)
        self.pq.fit(vector_list, max_iter=max_iter)

        # Encode all existing vectors into M-byte codes
        self.pq_codes = {
            doc_id: self.pq.encode(vec)
            for doc_id, vec in self.vectors.items()
        }
        self.use_pq = True

    def add(self, doc_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None, normalize: bool = False):
        """Adds a document vector to the store, updating PQ codes if enabled."""
        if normalize:
            mag = VectorMetrics.magnitude(vector)
            if mag == 0:
                raise ValueError("Cannot normalize a zero vector.")
            vector = [v / mag for v in vector]

        self.vectors[doc_id] = vector
        self.metadata[doc_id] = metadata or {}

        # Encode vector dynamically if PQ is fitted
        if self.pq is not None and self.pq.dim is not None:
            self.pq_codes[doc_id] = self.pq.encode(vector)

    def search_pq(self, query_vector: List[float], k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Performs fast O(M) Asymmetric Distance Computation (ADC) search using PQ codes."""
        if self.pq is None or not self.pq_codes:
            raise RuntimeError("PQ index is not trained. Call fit_pq() before searching with backend='pq'.")

        # Precompute 1x(M*K) distance lookup table for query
        lut = self.pq.compute_adc_table(query_vector)
        results = []

        for doc_id, code in self.pq_codes.items():
            # Apply metadata filters
            if filter_metadata:
                doc_meta = self.metadata.get(doc_id, {})
                if not all(doc_meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                    continue

            # Compute approximate Euclidean distance in O(M) operations
            dist = self.pq.adc_distance(lut, code)
            results.append({
                "id": doc_id,
                "score": dist,
                "metadata": self.metadata.get(doc_id, {})
            })

        # Sort ascending (lower distance is better)
        results.sort(key=lambda x: x["score"])
        return results[:k]

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        metric: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        backend: str = "exact"
    ) -> List[Dict[str, Any]]:
        """Unified search route supporting 'exact', 'hnsw', and 'pq' backends."""
        if backend == "pq":
            return self.search_pq(query_vector, k=k, filter_metadata=filter_metadata)
        elif backend == "hnsw":
            if not self.use_hnsw or self.hnsw_index is None:
                raise ValueError("HNSW backend is not initialized.")
            return self.hnsw_index.search(query_vector, k=k, filter_metadata=filter_metadata)
        elif backend == "exact":
            # Standard exact k-NN logic
            results = []
            for doc_id, vec in self.vectors.items():
                if filter_metadata:
                    doc_meta = self.metadata.get(doc_id, {})
                    if not all(doc_meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                        continue
                
                if metric == "cosine":
                    score = VectorMetrics.cosine_similarity(query_vector, vec)
                elif metric == "euclidean":
                    score = VectorMetrics.euclidean_distance(query_vector, vec)
                elif metric == "manhattan":
                    score = VectorMetrics.manhattan_distance(query_vector, vec)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                results.append({"id": doc_id, "score": score, "metadata": self.metadata.get(doc_id, {})})
            
            reverse_sort = (metric == "cosine")
            results.sort(key=lambda x: x["score"], reverse=reverse_sort)
            return results[:k]
        else:
            raise ValueError(f"Unsupported search backend: {backend}")