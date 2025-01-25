"""Tests for index persistence."""

import os
import tempfile

import numpy as np
import pytest

from vectordb.index import HNSWIndex
from vectordb.persistence import save_index, load_index


class TestPersistenceRoundtrip:
    def test_save_load_roundtrip(self):
        idx = HNSWIndex(dim=64, M=8, ef_construction=100)
        rng = np.random.RandomState(42)
        vecs = rng.randn(50, 64).astype(np.float64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            save_index(idx, tmp_path)
            loaded = load_index(tmp_path)
            assert loaded is not None
            assert loaded.dim == idx.dim
            assert loaded.metric == idx.metric
            assert loaded.element_count == idx.element_count
            assert loaded.entry_point == idx.entry_point

            # Search should work identically
            query = rng.randn(64).astype(np.float64)
            original = idx.search(query, k=5, ef=50)
            restored = loaded.search(query, k=5, ef=50)
            assert len(original) == len(restored)
            for (o_id, o_score), (r_id, r_score) in zip(original, restored):
                assert o_id == r_id
                assert o_score == pytest.approx(r_score)
        finally:
            os.unlink(tmp_path)

    def test_search_after_load(self):
        idx = HNSWIndex(dim=64, M=16, ef_construction=200)
        rng = np.random.RandomState(42)
        vecs = rng.randn(100, 64).astype(np.float64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            save_index(idx, tmp_path)
            loaded = load_index(tmp_path)
            assert loaded is not None

            query = vecs[0]
            results = loaded.search(query, k=10, ef=50)
            assert len(results) == 10
            # vecs[0] should be in top results (or very close)
            assert any(r[0] == "0" for r in results[:3])
        finally:
            os.unlink(tmp_path)

    def test_deleted_vectors_persist(self):
        idx = HNSWIndex(dim=64, M=8)
        rng = np.random.RandomState(42)
        vecs = rng.randn(20, 64).astype(np.float64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        idx.delete("0")
        idx.delete("5")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            save_index(idx, tmp_path)
            loaded = load_index(tmp_path)
            assert loaded is not None
            assert loaded.element_count == 18
            assert "0" in loaded.deleted
            assert "5" in loaded.deleted

            # Deleted items should not appear in search
            query = vecs[0]
            results = loaded.search(query, k=5, ef=50)
            assert all(r[0] != "0" for r in results)
        finally:
            os.unlink(tmp_path)


class TestPersistenceEdgeCases:
    def test_empty_index(self):
        idx = HNSWIndex(dim=64)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            save_index(idx, tmp_path)
            loaded = load_index(tmp_path)
            assert loaded is not None
            assert loaded.element_count == 0
            assert loaded.entry_point is None
        finally:
            os.unlink(tmp_path)

    def test_missing_file(self):
        result = load_index("/nonexistent/path/to/index.pkl")
        assert result is None

    def test_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"this is not a valid pickle file")
            tmp_path = f.name
        try:
            result = load_index(tmp_path)
            assert result is None
        finally:
            os.unlink(tmp_path)

    def test_all_vectors_deleted(self):
        idx = HNSWIndex(dim=64, M=8)
        rng = np.random.RandomState(42)
        vecs = rng.randn(10, 64).astype(np.float64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        for i in range(10):
            idx.delete(str(i))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            save_index(idx, tmp_path)
            loaded = load_index(tmp_path)
            assert loaded is not None
            assert loaded.element_count == 0
            results = loaded.search(vecs[0], k=1)
            assert results == []
        finally:
            os.unlink(tmp_path)
