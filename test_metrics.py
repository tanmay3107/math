import unittest
import math
from metrics import VectorMetrics

class TestVectorMetrics(unittest.TestCase):
    """Unit tests for the VectorMetrics utility class."""
    
    def setUp(self):
        self.vec_a = [1.0, 0.0, 0.0]
        self.vec_b = [0.0, 1.0, 0.0]
        self.vec_c = [1.0, 2.0, 3.0]
        self.vec_d = [4.0, 5.0, 6.0]

    def test_dot_product(self):
        self.assertEqual(VectorMetrics.dot_product(self.vec_a, self.vec_b), 0.0)
        self.assertEqual(VectorMetrics.dot_product(self.vec_c, self.vec_d), 32.0)

    def test_magnitude(self):
        self.assertEqual(VectorMetrics.magnitude(self.vec_a), 1.0)
        self.assertAlmostEqual(VectorMetrics.magnitude(self.vec_c), math.sqrt(14))

    def test_cosine_similarity(self):
        self.assertEqual(VectorMetrics.cosine_similarity(self.vec_a, self.vec_b), 0.0)
        self.assertAlmostEqual(VectorMetrics.cosine_similarity(self.vec_c, self.vec_c), 1.0)

    def test_euclidean_distance(self):
        # Distance between identical vectors is 0
        self.assertEqual(VectorMetrics.euclidean_distance(self.vec_a, self.vec_a), 0.0)
        # sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(27)
        self.assertAlmostEqual(VectorMetrics.euclidean_distance(self.vec_c, self.vec_d), math.sqrt(27))

    def test_manhattan_distance(self):
        # Distance between identical vectors is 0
        self.assertEqual(VectorMetrics.manhattan_distance(self.vec_a, self.vec_a), 0.0)
        # |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9.0
        self.assertEqual(VectorMetrics.manhattan_distance(self.vec_c, self.vec_d), 9.0)

    def test_length_mismatch_exception(self):
        # Check that mismatched vector lengths raise a ValueError
        short_vec = [1.0, 2.0]
        with self.assertRaises(ValueError):
            VectorMetrics.euclidean_distance(self.vec_c, short_vec)
        with self.assertRaises(ValueError):
            VectorMetrics.manhattan_distance(self.vec_c, short_vec)

    def test_zero_vector_exception(self):
        zero_vec = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            VectorMetrics.cosine_similarity(zero_vec, self.vec_a)

if __name__ == "__main__":
    unittest.main()