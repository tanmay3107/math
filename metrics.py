import math

class VectorMetrics:
    """A utility class for core vector distance and similarity calculations."""
    
    @staticmethod
    def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculates the dot product of two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of the same length.")
        return sum(a * b for a, b in zip(vec_a, vec_b))

    @staticmethod
    def magnitude(vec: list[float]) -> float:
        """Calculates the Euclidean magnitude (L2 norm) of a vector."""
        return math.sqrt(sum(v ** 2 for v in vec))

    @classmethod
    def cosine_similarity(cls, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        Returns a value between -1.0 and 1.0.
        """
        mag_a = cls.magnitude(vec_a)
        mag_b = cls.magnitude(vec_b)
        
        if mag_a == 0 or mag_b == 0:
            raise ValueError("Cannot calculate cosine similarity for a zero vector.")
            
        return cls.dot_product(vec_a, vec_b) / (mag_a * mag_b)

if __name__ == "__main__":
    # Quick sanity check
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    similarity = VectorMetrics.cosine_similarity(v1, v2)
    print(f"Cosine Similarity between {v1} and {v2}: {similarity:.4f}")import math

import math

class VectorMetrics:
    """A utility class for core vector distance and similarity calculations."""
    
    @staticmethod
    def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculates the dot product of two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of the same length.")
        return sum(a * b for a, b in zip(vec_a, vec_b))

    @staticmethod
    def magnitude(vec: list[float]) -> float:
        """Calculates the Euclidean magnitude (L2 norm) of a vector."""
        return math.sqrt(sum(v ** 2 for v in vec))

    @classmethod
    def cosine_similarity(cls, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        Returns a value between -1.0 and 1.0.
        """
        mag_a = cls.magnitude(vec_a)
        mag_b = cls.magnitude(vec_b)
        
        if mag_a == 0 or mag_b == 0:
            raise ValueError("Cannot calculate cosine similarity for a zero vector.")
            
        return cls.dot_product(vec_a, vec_b) / (mag_a * mag_b)

    @staticmethod
    def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculates the Euclidean distance (L2 norm distance) between two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of the same length.")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))

    @staticmethod
    def manhattan_distance(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculates the Manhattan distance (L1 norm distance) between two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of the same length.")
        return sum(abs(a - b) for a, b in zip(vec_a, vec_b))


if __name__ == "__main__":
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    
    print(f"Cosine Similarity: {VectorMetrics.cosine_similarity(v1, v2):.4f}")
    print(f"Euclidean Distance: {VectorMetrics.euclidean_distance(v1, v2):.4f}")
    print(f"Manhattan Distance: {VectorMetrics.manhattan_distance(v1, v2):.4f}")