import unittest
import math
from metrics import VectorMetrics

class TestVectorMetrics(unittest.TestCase):
    """Unit tests for the VectorMetrics utility class."""
    
    def setUp(self):
        # Set up a few standard vectors to reuse across tests
        self.vec_a = [1.0, 0.0, 0.0]
        self.vec_b = [0.0, 1.0, 0.0]
        self.vec_c = [1.0, 2.0, 3.0]
        self.vec_d = [4.0, 5.0, 6.0]

    def test_dot_product(self):
        # Orthogonal vectors should have a dot product of 0
        self.assertEqual(VectorMetrics.dot_product(self.vec_a, self.vec_b), 0.0)
        # (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        self.assertEqual(VectorMetrics.dot_product(self.vec_c, self.vec_d), 32.0)

    def test_magnitude(self):
        # Magnitude of a unit vector is 1
        self.assertEqual(VectorMetrics.magnitude(self.vec_a), 1.0)
        # sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
        self.assertAlmostEqual(VectorMetrics.magnitude(self.vec_c), math.sqrt(14))

    def test_cosine_similarity(self):
        # Orthogonal vectors share no similarity (0.0)
        self.assertEqual(VectorMetrics.cosine_similarity(self.vec_a, self.vec_b), 0.0)
        # Identical vectors are perfectly similar (1.0)
        self.assertAlmostEqual(VectorMetrics.cosine_similarity(self.vec_c, self.vec_c), 1.0)
        
    def test_zero_vector_exception(self):
        # Cosine similarity is undefined for zero vectors; check if error raises
        zero_vec = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            VectorMetrics.cosine_similarity(zero_vec, self.vec_a)

if __name__ == "__main__":
    unittest.main()