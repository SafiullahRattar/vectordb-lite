"""Tests for the HNSW graph index."""

import math
import numpy as np
import pytest

from vectordb.index import HNSWIndex


def _make_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float64)


class TestHNSWConstruction:
    def test_build_empty_index(self):
        idx = HNSWIndex(dim=128)
        assert idx.element_count == 0
        assert idx.entry_point is None

    def test_add_single_vector(self):
        idx = HNSWIndex(dim=128)
        idx.add(np.ones(128), "a")
        assert idx.element_count == 1
        assert idx.entry_point == "a"

    def test_add_multiple_vectors(self):
        idx = HNSWIndex(dim=128, M=8)
        vecs = _make_vectors(50, 128)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        assert idx.element_count == 50
        assert idx.entry_point is not None
        assert idx.entry_point in idx.vectors

    def test_layer_0_has_connections(self):
        idx = HNSWIndex(dim=128, M=8, ef_construction=50)
        vecs = _make_vectors(30, 128)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        # Every node should have at least one connection at layer 0
        for i in range(30):
            neighbors = idx._get_neighbors(str(i), 0)
            assert len(neighbors) >= 1, f"Node {i} has no layer-0 connections"


class TestLayerAssignment:
    def test_level_is_non_negative(self):
        idx = HNSWIndex(dim=64, M=16)
        levels = [idx._random_level() for _ in range(1000)]
        assert all(l >= 0 for l in levels)

    def test_exponential_decay(self):
        idx = HNSWIndex(dim=64, M=16)
        levels = [idx._random_level() for _ in range(5000)]
        # Most nodes should be at level 0
        level_0_count = sum(1 for l in levels if l == 0)
        assert level_0_count > 4000  # With M=16, P(level>=1)=1/16, so ~5000*15/16 ≈ 4687 at level 0
        # Very few should be at high levels
        level_3_plus = sum(1 for l in levels if l >= 3)
        assert level_3_plus < 10  # P(level>=3) = (1/16)^3 = 1/4096, so ~1 in 5000


class TestSearchCorrectness:
    def test_search_returns_inserted_vector(self):
        idx = HNSWIndex(dim=64, M=16, ef_construction=200)
        v = np.ones(64)
        idx.add(v, "a")
        results = idx.search(v, k=1, ef=10)
        assert len(results) == 1
        assert results[0][0] == "a"
        # Score should be near 1.0 for cosine with itself
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_search_returns_k_results(self):
        idx = HNSWIndex(dim=64, M=16, ef_construction=200)
        vecs = _make_vectors(100, 64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        results = idx.search(vecs[0], k=5, ef=50)
        assert len(results) == 5

    def test_search_finds_closest(self):
        idx = HNSWIndex(dim=64, metric="euclidean", M=16, ef_construction=200)
        # Insert known vectors
        idx.add(np.array([0.0, 0.0] + [0.0] * 62), "origin")
        idx.add(np.array([1.0, 0.0] + [0.0] * 62), "x1")
        idx.add(np.array([10.0, 0.0] + [0.0] * 62), "x10")
        # Query near origin
        results = idx.search(np.array([0.0, 0.1] + [0.0] * 62), k=2, ef=10)
        # origin should be closest
        assert results[0][0] == "origin"

    def test_search_does_not_return_deleted(self):
        idx = HNSWIndex(dim=64, M=16, ef_construction=200)
        idx.add(np.ones(64), "a")
        idx.add(-np.ones(64), "b")
        idx.delete("a")
        results = idx.search(np.ones(64), k=1, ef=10)
        assert len(results) == 1
        assert results[0][0] == "b"

    def test_search_empty_index(self):
        idx = HNSWIndex(dim=64)
        results = idx.search(np.ones(64), k=5)
        assert results == []


class TestEdgeConsistency:
    def test_bidirectional_edges(self):
        idx = HNSWIndex(dim=64, M=8, ef_construction=100)
        vecs = _make_vectors(20, 64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        # Check all edges are bidirectional at all levels
        for level, node_dict in idx.layers.items():
            for node_id, neighbors in node_dict.items():
                for nb in neighbors:
                    assert node_id in idx.layers[level].get(nb, set()), (
                        f"Edge {node_id} -> {nb} at level {level} is not bidirectional"
                    )

    def test_no_self_loops(self):
        idx = HNSWIndex(dim=64, M=8, ef_construction=100)
        vecs = _make_vectors(20, 64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        for level, node_dict in idx.layers.items():
            for node_id, neighbors in node_dict.items():
                assert node_id not in neighbors, f"Self-loop at {node_id} level {level}"


class TestDeleteAndRepair:
    def test_entry_point_repair_on_delete(self):
        idx = HNSWIndex(dim=64, M=8, ef_construction=100)
        vecs = _make_vectors(10, 64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        old_ep = idx.entry_point
        idx.delete(old_ep)
        assert idx.entry_point != old_ep
        assert idx.entry_point is not None
        assert idx.entry_point not in idx.deleted

    def test_delete_reduces_element_count(self):
        idx = HNSWIndex(dim=64, M=8)
        vecs = _make_vectors(5, 64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        assert idx.element_count == 5
        idx.delete("0")
        assert idx.element_count == 4

    def test_delete_all_then_search(self):
        idx = HNSWIndex(dim=64, M=8)
        vecs = _make_vectors(5, 64)
        for i, v in enumerate(vecs):
            idx.add(v, str(i))
        for i in range(5):
            idx.delete(str(i))
        assert idx.element_count == 0
        results = idx.search(vecs[0], k=1)
        assert results == []

    def test_re_add_deleted(self):
        idx = HNSWIndex(dim=64, M=8)
        v = np.ones(64)
        idx.add(v, "a")
        idx.delete("a")
        assert idx.element_count == 0
        idx.add(v, "a")
        assert idx.element_count == 1
        results = idx.search(v, k=1)
        assert len(results) == 1
        assert results[0][0] == "a"
