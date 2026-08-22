import random
import math
import heapq
from metrics import VectorMetrics

class HNSWIndex:
    """Hierarchical Navigable Small World (HNSW) graph index for fast Approximate Nearest Neighbor (ANN) search."""

    def __init__(self, distance_metric="cosine", M=16, ef_construction=64, ef_search=32):
        self.metric = distance_metric
        self.M = M                           # Max neighbors per node in layers > 0
        self.M0 = 2 * M                      # Max neighbors per node in layer 0
        self.ef_construction = ef_construction # Beam search width during build
        self.ef_search = ef_search           # Beam search width during query
        self.mL = 1.0 / math.log(M)          # Normalization factor for level generation

        self.vectors = {}                    # node_id -> vector
        self.graphs = []                     # list of layer dicts: level -> {node_id: set(neighbor_ids)}
        self.enter_node = None               # Global entry point node ID
        self.max_level = -1

    def _distance(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute distance (lower is closer)."""
        if self.metric == "cosine":
            return 1.0 - VectorMetrics.cosine_similarity(vec1, vec2)
        elif self.metric == "euclidean":
            return VectorMetrics.euclidean_distance(vec1, vec2)
        elif self.metric == "manhattan":
            return VectorMetrics.manhattan_distance(vec1, vec2)
        else:
            raise ValueError(f"Unsupported metric '{self.metric}'.")

    def _random_level(self) -> int:
        """Generate a random layer index using exponential decay."""
        return int(-math.log(random.random()) * self.mL)

    def _search_layer(self, query: list[float], entry_points: list[str], ef: int, level: int) -> list[tuple[float, str]]:
        """Beam search within a single graph layer."""
        visited = set(entry_points)
        candidates = []  # Min-heap for exploration: (dist, node_id)
        w_results = []   # Max-heap for top ef results: (-dist, node_id)

        for ep in entry_points:
            dist = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w_results, (-dist, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            furthest_dist = -w_results[0][0]

            if dist_c > furthest_dist:
                break

            for neighbor in self.graphs[level].get(c, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist_n = self._distance(query, self.vectors[neighbor])
                    furthest_dist = -w_results[0][0]

                    if dist_n < furthest_dist or len(w_results) < ef:
                        heapq.heappush(candidates, (dist_n, neighbor))
                        heapq.heappush(w_results, (-dist_n, neighbor))
                        if len(w_results) > ef:
                            heapq.heappop(w_results)

        results = [(-neg_d, node_id) for neg_d, node_id in w_results]
        results.sort(key=lambda x: x[0])
        return results

    def add(self, node_id: str, vector: list[float]):
        """Insert a new vector into the multi-layer HNSW graph structure."""
        if node_id in self.vectors:
            return

        self.vectors[node_id] = vector
        level = self._random_level()

        # Ensure graph layers exist up to the generated level
        while len(self.graphs) <= level:
            self.graphs.append({})

        curr_entry = self.enter_node

        if curr_entry is None:
            # First node inserted into empty index
            for l in range(level + 1):
                self.graphs[l][node_id] = set()
            self.enter_node = node_id
            self.max_level = level
            return

        # Phase 1: Greedily descend from max_level down to insertion level + 1
        curr_obj = curr_entry
        curr_dist = self._distance(vector, self.vectors[curr_obj])

        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, set()):
                    d = self._distance(vector, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Phase 2: Connect neighbors from min(level, max_level) down to Layer 0
        ep = [curr_obj]
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, l)
            ep = [c[1] for c in candidates]

            max_m = self.M0 if l == 0 else self.M
            neighbors = ep[:max_m]

            self.graphs[l][node_id] = set(neighbors)
            for neighbor in neighbors:
                if neighbor not in self.graphs[l]:
                    self.graphs[l][neighbor] = set()
                self.graphs[l][neighbor].add(node_id)

                # Prune connections if node exceeds max degree capacity
                if len(self.graphs[l][neighbor]) > max_m:
                    pruned = sorted(
                        list(self.graphs[l][neighbor]),
                        key=lambda nid: self._distance(self.vectors[neighbor], self.vectors[nid])
                    )[:max_m]
                    self.graphs[l][neighbor] = set(pruned)

        if level > self.max_level:
            self.enter_node = node_id
            self.max_level = level

    def search(self, query_vector: list[float], k: int = 5, ef: int = None) -> list[dict]:
        """Search top-k approximate nearest neighbors."""
        if self.enter_node is None:
            return []

        ef_search = max(ef or self.ef_search, k)
        curr_obj = self.enter_node
        curr_dist = self._distance(query_vector, self.vectors[curr_obj])

        # Greedily navigate top layers down to Layer 1
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, set()):
                    d = self._distance(query_vector, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Beam search at Layer 0 for candidates
        candidates = self._search_layer(query_vector, [curr_obj], ef_search, 0)
        
        results = []
        for dist, node_id in candidates[:k]:
            score = (1.0 - dist) if self.metric == "cosine" else dist
            results.append({
                "id": node_id,
                "score": score,
                "vector": self.vectors[node_id]
            })

        return results