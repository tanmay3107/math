import math
import json
import re
from collections import defaultdict
from metrics import VectorMetrics

class VectorStore:
    """Vector database with metadata filtering, persistence, CRUD, and hybrid (keyword + vector) search."""

    def __init__(self):
        self.vectors = {}         # id -> vector list
        self.metadata = {}        # id -> metadata dict
        self.inverted_index = defaultdict(set)  # term -> set of doc_ids
        self.doc_term_freqs = defaultdict(lambda: defaultdict(int)) # doc_id -> {term: count}
        self.doc_lengths = {}     # doc_id -> total token count

    def _tokenize(self, text: str) -> list[str]:
        """Convert string to lowercased alphanumeric tokens."""
        if not isinstance(text, str):
            return []
        return re.findall(r'\w+', text.lower())

    def _index_doc(self, doc_id: str, metadata: dict):
        """Index all string values in metadata for keyword search."""
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
        """Remove document terms from the inverted index."""
        if doc_id in self.doc_term_freqs:
            for token in list(self.doc_term_freqs[doc_id].keys()):
                self.inverted_index[token].discard(doc_id)
                if not self.inverted_index[token]:
                    del self.inverted_index[token]
            del self.doc_term_freqs[doc_id]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]

    def add(self, doc_id: str, vector: list[float], metadata: dict = None, normalize: bool = False):
        """Add or overwrite a vector entry and index its metadata."""
        if normalize:
            vector = VectorMetrics.normalize(vector)
        self.vectors[doc_id] = vector
        self.metadata[doc_id] = metadata or {}
        self._index_doc(doc_id, self.metadata[doc_id])

    def add_batch(self, records: list[dict], normalize: bool = False):
        """Add a batch of records: [{'id': str, 'vector': list, 'metadata': dict}]."""
        for rec in records:
            if "id" not in rec or "vector" not in rec:
                raise KeyError("Each record must contain 'id' and 'vector' keys.")
            self.add(rec["id"], rec["vector"], rec.get("metadata"), normalize=normalize)

    def get(self, doc_id: str) -> dict | None:
        """Retrieve a record by ID."""
        if doc_id not in self.vectors:
            return None
        return {
            "id": doc_id,
            "vector": self.vectors[doc_id],
            "metadata": self.metadata[doc_id]
        }

    def delete(self, doc_id: str) -> bool:
        """Delete a vector and its indexed terms."""
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.metadata[doc_id]
            self._unindex_doc(doc_id)
            return True
        return False

    def update(self, doc_id: str, vector: list[float] = None, metadata: dict = None, normalize: bool = False) -> bool:
        """Partially or fully update a vector and its metadata."""
        if doc_id not in self.vectors:
            return False
        if vector is not None:
            if normalize:
                vector = VectorMetrics.normalize(vector)
            self.vectors[doc_id] = vector
        if metadata is not None:
            self.metadata[doc_id] = metadata
            self._index_doc(doc_id, metadata)
        return True

    def search(self, query_vector: list[float], k: int = 5, metric: str = "cosine", filter_metadata: dict = None) -> list[dict]:
        """Perform dense vector similarity search."""
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
        """Perform BM25-style term frequency keyword search over indexed metadata."""
        tokens = self._tokenize(query_text)
        if not tokens or not self.vectors:
            return []

        num_docs = len(self.vectors)
        avg_dl = sum(self.doc_lengths.values()) / num_docs if num_docs > 0 else 1.0
        scores = defaultdict(float)

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        for token in tokens:
            matching_docs = self.inverted_index.get(token, set())
            doc_freq = len(matching_docs)
            if doc_freq == 0:
                continue

            # Inverse Document Frequency (IDF)
            idf = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            for doc_id in matching_docs:
                tf = self.doc_term_freqs[doc_id][token]
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 term weight
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

    def hybrid_search(self, query_vector: list[float], query_text: str, k: int = 5, rrf_k: int = 60, metric: str = "cosine") -> list[dict]:
        """Hybrid search combining vector and keyword results via Reciprocal Rank Fusion (RRF)."""
        vector_res = self.search(query_vector, k=k*2, metric=metric)
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
        """Serialize state to JSON file."""
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
        """Load state from JSON file and rebuild inverted index."""
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
    def from_json(cls, filepath: str):
        """Factory method to construct instance directly from JSON file."""
        store = cls()
        store.load_from_json(filepath)
        return store