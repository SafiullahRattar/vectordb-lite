"""Tests for similarity computation functions."""

import numpy as np
import pytest

from vectordb.similarity import cosine_similarity, euclidean_distance, dot_product


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_known_value(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.5, np.sqrt(3) / 2])
        assert cosine_similarity(a, b) == pytest.approx(0.5)

    def test_2d_arrays(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = cosine_similarity(a, b)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        result = cosine_similarity(a, b)
        assert np.isfinite(result)


class TestEuclideanDistance:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_symmetry(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_2d_arrays(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        b = np.array([[3.0, 4.0], [0.0, 0.0]])
        result = euclidean_distance(a, b)
        assert len(result) == 2
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(5.0)


class TestDotProduct:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert dot_product(a, a) == pytest.approx(14.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert dot_product(a, b) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert dot_product(a, b) == pytest.approx(11.0)

    def test_2d_arrays(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = dot_product(a, b)
        assert len(result) == 2
        assert result[0] == pytest.approx(17.0)
        assert result[1] == pytest.approx(53.0)
