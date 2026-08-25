import math
import json
import re
from collections import defaultdict
from metrics import VectorMetrics
from hnsw import HNSWIndex

class VectorStore:
    """Vector database with metadata filtering, persistence, CRUD, hybrid search,
    and configurable HNSW ANN search backend.
    """

    def __init__(self, use_hnsw: bool = False, hnsw_kwargs: dict = None):
        self.vectors = {}         # id -> vector list
        self.metadata = {}        # id -> metadata dict
        self.inverted_index = defaultdict(set)  # term -> set of doc_ids
        self.doc_term_freqs = defaultdict(lambda: defaultdict(int)) # doc_id -> {term: count}
        self.doc_lengths = {}     # doc_id -> total token count

        self.use_hnsw = use_hnsw
        self.hnsw_kwargs = hnsw_kwargs or {"distance_metric": "cosine", "M": 16, "ef_construction": 64, "ef_search": 32}
        self.hnsw_index = HNSWIndex(**self.hnsw_kwargs) if use_hnsw else None

    def _tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        return re.findall(r'\w+', text.lower())

    def _index_doc(self, doc_id: str, metadata: dict):
        self._unindex_doc(doc_id)
        if not metadata:
            return

        all_text = " ".join([str(v) for v in metadata.values() if isinstance(v, (str, list, dict))])
        tokens = self._tokenize(all_text)
        
        if not tokens:
            return

        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.inverted_index[token].add(doc_id)
            self.doc_term_freqs[doc_id][token] += 1

    def _unindex_doc(self, doc_id: str):
        if doc_id in self.doc_term_freqs:
            for token in list(self.doc_term_freqs[doc_id].keys()):
                self.inverted_index[token].discard(doc_id)
                if not self.inverted_index[token]:
                    del self.inverted_index[token]
            del self.doc_term_freqs[doc_id]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]

    def _rebuild_hnsw(self):
        """Rebuild HNSW index from scratch after deletions or bulk reloads."""
        if not self.use_hnsw:
            return
        self.hnsw_index = HNSWIndex(**self.hnsw_kwargs)
        for doc_id, vec in self.vectors.items():
            self.hnsw_index.add(doc_id, vec)

    def add(self, doc_id: str, vector: list[float], metadata: dict = None, normalize: bool = False):
        if normalize:
            vector = VectorMetrics.normalize(vector)
        self.vectors[doc_id] = vector
        self.metadata[doc_id] = metadata or {}
        self._index_doc(doc_id, self.metadata[doc_id])

        if self.use_hnsw:
            self.hnsw_index.add(doc_id, vector)

    def add_batch(self, records: list[dict], normalize: bool = False):
        for rec in records:
            if "id" not in rec or "vector" not in rec:
                raise KeyError("Each record must contain 'id' and 'vector' keys.")
            self.add(rec["id"], rec["vector"], rec.get("metadata"), normalize=normalize)

    def get(self, doc_id: str) -> dict | None:
        if doc_id not in self.vectors:
            return None
        return {
            "id": doc_id,
            "vector": self.vectors[doc_id],
            "metadata": self.metadata[doc_id]
        }

    def delete(self, doc_id: str) -> bool:
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.metadata[doc_id]
            self._unindex_doc(doc_id)
            if self.use_hnsw:
                self._rebuild_hnsw()
            return True
        return False

    def update(self, doc_id: str, vector: list[float] = None, metadata: dict = None, normalize: bool = False) -> bool:
        if doc_id not in self.vectors:
            return False
        if vector is not None:
            if normalize:
                vector = VectorMetrics.normalize(vector)
            self.vectors[doc_id] = vector
        if metadata is not None:
            self.metadata[doc_id] = metadata
            self._index_doc(doc_id, metadata)
        
        if self.use_hnsw and vector is not None:
            self._rebuild_hnsw()
        return True

    def search(self, query_vector: list[float], k: int = 5, metric: str = "cosine", 
               filter_metadata: dict = None, backend: str = "exact") -> list[dict]:
        """Perform vector search using either 'exact' brute-force or 'hnsw' graph backend."""
        if backend == "hnsw":
            if not self.use_hnsw:
                raise ValueError("HNSW backend is not enabled on this VectorStore instance.")
            
            # Fetch extra candidates from HNSW to allow for metadata post-filtering
            fetch_k = k * 5 if filter_metadata else k
            raw_results = self.hnsw_index.search(query_vector, k=fetch_k)
            
            results = []
            for item in raw_results:
                doc_id = item["id"]
                if filter_metadata:
                    doc_meta = self.metadata.get(doc_id, {})
                    if not all(doc_meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                        continue

                results.append({
                    "id": doc_id,
                    "score": item["score"],
                    "vector": self.vectors[doc_id],
                    "metadata": self.metadata.get(doc_id, {})
                })
                if len(results) == k:
                    break
            return results

        # Default: Exact brute-force search
        results = []
        metric_fn = getattr(VectorMetrics, f"{metric}_similarity", None)
        if metric_fn is None:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine', 'euclidean', or 'manhattan'.")

        for doc_id, vector in self.vectors.items():
            if filter_metadata:
                doc_meta = self.metadata.get(doc_id, {})
                if not all(doc_meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                    continue

            score = metric_fn(query_vector, vector)
            results.append({
                "id": doc_id,
                "score": score,
                "vector": vector,
                "metadata": self.metadata[doc_id]
            })

        reverse_sort = True if metric == "cosine" else False
        results.sort(key=lambda x: x["score"], reverse=reverse_sort)
        return results[:k]

    def keyword_search(self, query_text: str, k: int = 5) -> list[dict]:
        tokens = self._tokenize(query_text)
        if not tokens or not self.vectors:
            return []

        num_docs = len(self.vectors)
        avg_dl = sum(self.doc_lengths.values()) / num_docs if num_docs > 0 else 1.0
        scores = defaultdict(float)

        k1, b = 1.5, 0.75

        for token in tokens:
            matching_docs = self.inverted_index.get(token, set())
            doc_freq = len(matching_docs)
            if doc_freq == 0:
                continue

            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            for doc_id in matching_docs:
                tf = self.doc_term_freqs[doc_id][token]
                doc_len = self.doc_lengths[doc_id]
                denom = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
                term_score = idf * (tf * (k1 + 1.0)) / denom
                scores[doc_id] += term_score

        results = [
            {
                "id": doc_id,
                "score": score,
                "vector": self.vectors[doc_id],
                "metadata": self.metadata[doc_id]
            }
            for doc_id, score in scores.items()
        ]

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def hybrid_search(self, query_vector: list[float], query_text: str, k: int = 5, 
                      rrf_k: int = 60, metric: str = "cosine", backend: str = "exact") -> list[dict]:
        """Hybrid search combining vector search (exact or HNSW) and keyword search via RRF."""
        vector_res = self.search(query_vector, k=k*2, metric=metric, backend=backend)
        keyword_res = self.keyword_search(query_text, k=k*2)

        rrf_scores = defaultdict(float)

        for rank, res in enumerate(vector_res, start=1):
            rrf_scores[res["id"]] += 1.0 / (rrf_k + rank)

        for rank, res in enumerate(keyword_res, start=1):
            rrf_scores[res["id"]] += 1.0 / (rrf_k + rank)

        combined_results = []
        for doc_id, score in rrf_scores.items():
            combined_results.append({
                "id": doc_id,
                "rrf_score": score,
                "vector": self.vectors[doc_id],
                "metadata": self.metadata[doc_id]
            })

        combined_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return combined_results[:k]

    def save_to_json(self, filepath: str):
        data = {
            doc_id: {
                "vector": self.vectors[doc_id],
                "metadata": self.metadata[doc_id]
            }
            for doc_id in self.vectors
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.vectors.clear()
        self.metadata.clear()
        self.inverted_index.clear()
        self.doc_term_freqs.clear()
        self.doc_lengths.clear()

        for doc_id, payload in data.items():
            self.add(doc_id, payload["vector"], payload.get("metadata"))

    @classmethod
    def from_json(cls, filepath: str, use_hnsw: bool = False, hnsw_kwargs: dict = None):
        store = cls(use_hnsw=use_hnsw, hnsw_kwargs=hnsw_kwargs)
        store.load_from_json(filepath)
        return store