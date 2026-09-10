import math
import random
from typing import List, Tuple, Optional


def _euclidean_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


class ProductQuantizer:
    """
    Product Quantization (PQ) vector compression engine.
    Compresses D-dimensional float32 vectors into M bytes (8x-32x memory compression).
    """

    def __init__(self, num_subvectors: int = 8, num_centroids: int = 256):
        self.M = num_subvectors
        self.K = num_centroids  # 256 centroids = 1 byte (uint8) per sub-vector
        self.dim: Optional[int] = None
        self.sub_dim: Optional[int] = None
        # Centroids storage shape: [M, K, sub_dim]
        self.centroids: List[List[List[float]]] = []

    def fit(self, vectors: List[List[float]], max_iter: int = 20):
        """Learns sub-space centroids using Lloyd's K-Means algorithm."""
        if not vectors:
            raise ValueError("Training dataset cannot be empty.")

        self.dim = len(vectors[0])
        if self.dim % self.M != 0:
            raise ValueError(f"Vector dimension {self.dim} must be divisible by num_subvectors {self.M}")

        self.sub_dim = self.dim // self.M
        self.centroids = []

        # Train K-Means independently for each of the M sub-vector spaces
        for m in range(self.M):
            sub_vectors = [v[m * self.sub_dim : (m + 1) * self.sub_dim] for v in vectors]
            m_centroids = self._kmeans(sub_vectors, self.K, max_iter)
            self.centroids.append(m_centroids)

    def _kmeans(self, data: List[List[float]], k: int, max_iter: int) -> List[List[float]]:
        """Simple, zero-dependency K-Means clustering for sub-vectors."""
        n_samples = len(data)
        k = min(k, n_samples)
        
        # Initialize centroids randomly from data samples
        random.seed(42)
        centroid_indices = random.sample(range(n_samples), k)
        centroids = [data[i][:] for i in centroid_indices]

        for _ in range(max_iter):
            # Assignment phase
            clusters: List[List[List[float]]] = [[] for _ in range(k)]
            for vec in data:
                best_idx = min(range(k), key=lambda i: _euclidean_sq(vec, centroids[i]))
                clusters[best_idx].append(vec)

            # Update phase
            new_centroids = []
            for i in range(k):
                if not clusters[i]:
                    new_centroids.append(centroids[i])
                    continue
                mean_vec = [
                    sum(c[d] for c in clusters[i]) / len(clusters[i])
                    for d in range(self.sub_dim)
                ]
                new_centroids.append(mean_vec)

            if new_centroids == centroids:
                break
            centroids = new_centroids

        return centroids

    def encode(self, vector: List[float]) -> List[int]:
        """Compresses a D-dimensional float vector into M byte codes."""
        if self.dim is None:
            raise RuntimeError("Quantizer must be fitted before encoding.")

        code = []
        for m in range(self.M):
            sub_vec = vector[m * self.sub_dim : (m + 1) * self.sub_dim]
            best_code = min(
                range(len(self.centroids[m])),
                key=lambda k: _euclidean_sq(sub_vec, self.centroids[m][k]),
            )
            code.append(best_code)
        return code

    def decode(self, code: List[int]) -> List[float]:
        """Reconstructs approximate float vector from M byte codes."""
        reconstructed = []
        for m, c in enumerate(code):
            reconstructed.extend(self.centroids[m][c])
        return reconstructed

    def compute_adc_table(self, query: List[float]) -> List[List[float]]:
        """
        Precomputes Asymmetric Distance Computation (ADC) Lookup Table (LUT).
        LUT shape: [M, K] containing squared distance from query sub-vector to each centroid.
        """
        lut = []
        for m in range(self.M):
            q_sub = query[m * self.sub_dim : (m + 1) * self.sub_dim]
            m_distances = [
                _euclidean_sq(q_sub, self.centroids[m][k])
                for k in range(len(self.centroids[m]))
            ]
            lut.append(m_distances)
        return lut

    def adc_distance(self, lut: List[List[float]], code: List[int]) -> float:
        """Computes approximate distance in O(M) lookup operations using precomputed ADC table."""
        return math.sqrt(sum(lut[m][code[m]] for m in range(self.M)))


if __name__ == "__main__":
    import random

    # Generate synthetic 128-dim vectors
    DIM, M, NUM_VECS = 128, 8, 500
    training_data = [[random.gauss(0, 1) for _ in range(DIM)] for _ in range(NUM_VECS)]

    pq = ProductQuantizer(num_subvectors=M, num_centroids=256)
    pq.fit(training_data)

    test_vec = training_data[0]
    encoded_code = pq.encode(test_vec)
    decoded_vec = pq.decode(encoded_code)

    lut = pq.compute_adc_table(test_vec)
    approx_dist = pq.adc_distance(lut, encoded_code)

    orig_bytes = DIM * 4
    compressed_bytes = M * 1

    print(f"Original vector size : {orig_bytes} bytes")
    print(f"Compressed code size : {compressed_bytes} bytes ({orig_bytes / compressed_bytes:.1f}x compression)")
    print(f"ADC Estimated Dist   : {approx_dist:.4f}")