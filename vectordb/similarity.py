"""Similarity computation functions for vector comparison.

All functions support both 1D and 2D numpy arrays. For 1D arrays,
a scalar is returned. For 2D arrays, pairwise computations are
performed and a 1D array of results is returned.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector (1D) or array of vectors (2D).
    b : np.ndarray
        Second vector (1D) or array of vectors (2D).
        Must match shape of ``a``.

    Returns
    -------
    float
        Cosine similarity. For 1D inputs, returns a scalar in [-1, 1].
        For 2D inputs, returns a 1D array of pairwise similarities.
    """
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1e-10, a_norm)
    b_norm = np.where(b_norm == 0, 1e-10, b_norm)
    a_unit = a / a_norm
    b_unit = b / b_norm
    result = np.sum(a_unit * b_unit, axis=-1)
    if a.ndim == 1:
        return float(result)
    return result


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector (1D) or array of vectors (2D).
    b : np.ndarray
        Second vector (1D) or array of vectors (2D).
        Must match shape of ``a``.

    Returns
    -------
    float
        Euclidean distance. For 1D inputs, returns a scalar >= 0.
        For 2D inputs, returns a 1D array of pairwise distances.
    """
    diff = a - b
    result = np.sqrt(np.sum(diff * diff, axis=-1))
    if a.ndim == 1:
        return float(result)
    return result


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector (1D) or array of vectors (2D).
    b : np.ndarray
        Second vector (1D) or array of vectors (2D).
        Must match shape of ``a``.

    Returns
    -------
    float
        Dot product. For 1D inputs, returns a scalar.
        For 2D inputs, returns a 1D array of pairwise products.
    """
    result = np.sum(a * b, axis=-1)
    if a.ndim == 1:
        return float(result)
    return result
