"""Integration tests for the public VectorDB API."""

import os
import tempfile

import numpy as np
import pytest

from vectordb.api import VectorDB


def _make_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float64)


class TestVectorDBAdd:
    def test_add_single(self):
        db = VectorDB(dim=64)
        db.add(np.ones(64), ["a"])
        assert len(db) == 1

    def test_add_batch(self):
        db = VectorDB(dim=64)
        vecs = _make_vectors(100, 64)
        ids = [str(i) for i in range(100)]
        db.add(vecs, ids)
        assert len(db) == 100

    def test_add_wrong_dimension(self):
        db = VectorDB(dim=64)
        with pytest.raises(ValueError):
            db.add(np.ones(32), ["a"])

    def test_add_mismatched_ids(self):
        db = VectorDB(dim=64)
        with pytest.raises(ValueError):
            db.add(_make_vectors(5, 64), ["a", "b"])


class TestVectorDBSearch:
    def test_search_basic(self):
        db = VectorDB(dim=64, metric="cosine")
        vecs = _make_vectors(200, 64)
        ids = [str(i) for i in range(200)]
        db.add(vecs, ids)

        results = db.search(vecs[0], k=5)
        assert len(results) == 5
        assert results[0][0] == "0"  # vector 0 should be closest to itself

    def test_search_returns_scores(self):
        db = VectorDB(dim=64)
        vecs = _make_vectors(100, 64)
        ids = [str(i) for i in range(100)]
        db.add(vecs, ids)

        results = db.search(vecs[0], k=5)
        for _id, score in results:
            assert isinstance(_id, str)
            assert isinstance(score, float)

    def test_search_empty(self):
        db = VectorDB(dim=64)
        results = db.search(np.ones(64), k=5)
        assert results == []

    def test_search_invalid_query_dim(self):
        db = VectorDB(dim=64)
        vecs = _make_vectors(10, 64)
        db.add(vecs, [str(i) for i in range(10)])
        with pytest.raises(ValueError):
            db.search(np.ones(32), k=5)

    def test_search_euclidean_metric(self):
        db = VectorDB(dim=64, metric="euclidean")
        vecs = _make_vectors(100, 64)
        ids = [str(i) for i in range(100)]
        db.add(vecs, ids)

        results = db.search(vecs[0], k=5)
        assert len(results) == 5
        assert results[0][0] == "0"

    def test_search_dot_product_metric(self):
        db = VectorDB(dim=64, metric="dot_product")
        vecs = _make_vectors(100, 64)
        ids = [str(i) for i in range(100)]
        db.add(vecs, ids)

        results = db.search(vecs[0], k=5)
        assert len(results) == 5


class TestVectorDBDelete:
    def test_delete_basic(self):
        db = VectorDB(dim=64)
        vecs = _make_vectors(10, 64)
        ids = [str(i) for i in range(10)]
        db.add(vecs, ids)
        db.delete("0")
        assert len(db) == 9

    def test_delete_not_found(self):
        db = VectorDB(dim=64)
        db.delete("nonexistent")
        assert len(db) == 0

    def test_delete_affects_search(self):
        db = VectorDB(dim=64)
        vecs = _make_vectors(50, 64)
        ids = [str(i) for i in range(50)]
        db.add(vecs, ids)

        db.delete("0")
        results = db.search(vecs[0], k=10)
        assert "0" not in [r[0] for r in results]


class TestVectorDBPersistence:
    def test_save_load_roundtrip(self):
        db = VectorDB(dim=64, metric="cosine", M=16, ef_construction=200)
        vecs = _make_vectors(200, 64)
        ids = [str(i) for i in range(200)]
        db.add(vecs, ids)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            db.save(tmp_path)
            restored = VectorDB.load(tmp_path)
            assert restored is not None
            assert len(restored) == 200

            results_orig = db.search(vecs[0], k=5)
            results_rest = restored.search(vecs[0], k=5)
            assert [r[0] for r in results_orig] == [r[0] for r in results_rest]
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent(self):
        restored = VectorDB.load("/nonexistent/file.pkl")
        assert restored is None

    def test_save_empty(self):
        db = VectorDB(dim=64)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            db.save(tmp_path)
            restored = VectorDB.load(tmp_path)
            assert restored is not None
            assert len(restored) == 0
            assert restored.search(np.ones(64)) == []
        finally:
            os.unlink(tmp_path)


class TestVectorDBLen:
    def test_len_zero(self):
        db = VectorDB(dim=64)
        assert len(db) == 0

    def test_len_after_add(self):
        db = VectorDB(dim=64)
        db.add(_make_vectors(5, 64), ["a", "b", "c", "d", "e"])
        assert len(db) == 5

    def test_len_after_delete(self):
        db = VectorDB(dim=64)
        db.add(_make_vectors(5, 64), ["a", "b", "c", "d", "e"])
        db.delete("c")
        assert len(db) == 4

    def test_len_after_save_load(self):
        db = VectorDB(dim=64)
        db.add(_make_vectors(10, 64), [str(i) for i in range(10)])
        db.delete("0")
        db.delete("1")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            db.save(tmp_path)
            restored = VectorDB.load(tmp_path)
            assert restored is not None
            assert len(restored) == 8
        finally:
            os.unlink(tmp_path)
