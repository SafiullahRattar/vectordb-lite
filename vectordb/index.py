"""HNSW graph index implementation.

Provides an in-memory Hierarchical Navigable Small World (HNSW) graph
for approximate nearest neighbor search with configurable metrics,
beam search, heuristic neighbor selection, and soft deletion.
"""

import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class HNSWIndex:
    """Hierarchical Navigable Small World graph index.

    Implements multi-layer graph construction and search following the
    HNSW algorithm. Supports cosine similarity, Euclidean distance,
    and dot product as distance metrics.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors stored in the index.
    metric : str, optional
        Distance metric. One of ``"cosine"``, ``"euclidean"``, or
        ``"dot_product"``. Default is ``"cosine"``.
    M : int, optional
        Maximum number of connections per node in higher layers.
        Layer 0 uses ``2 * M`` connections. Default is 16.
    ef_construction : int, optional
        Beam search width during index construction. Larger values
        produce higher-quality graphs at the cost of build time.
        Default is 200.

    Attributes
    ----------
    element_count : int
        Number of active (non-deleted) vectors in the index.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
    ) -> None:
        if M <= 0:
            raise ValueError("M must be positive")
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M_max = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ml = 1.0 / math.log(max(M, 2))

        self.vectors: Dict[str, np.ndarray] = {}
        self.node_levels: Dict[str, int] = {}
        self.entry_point: Optional[str] = None
        self.deleted: Set[str] = set()
        self.layers: Dict[int, Dict[str, Set[str]]] = {}

    @property
    def element_count(self) -> int:
        """Number of active (non-deleted) vectors."""
        return len(self.vectors) - len(self.deleted)

    # ---- Distance helpers ------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute internal distance (smaller means closer)."""
        from vectordb.similarity import cosine_similarity, dot_product, euclidean_distance

        if self.metric == "cosine":
            return float(1.0 - cosine_similarity(a, b))
        elif self.metric == "euclidean":
            return float(euclidean_distance(a, b))
        elif self.metric == "dot_product":
            return float(-dot_product(a, b))
        raise ValueError(f"Unknown metric: {self.metric}")

    # ---- Level assignment ------------------------------------------------

    def _random_level(self) -> int:
        """Generate a random level with exponential decay.

        Uses ``level = floor(-ln(uniform(0, 1)) * mL)`` where
        ``mL = 1 / ln(M)``.  This gives ``P(level >= l) = (1/M)^l``.
        """
        r = random.random()
        if r == 0.0:
            r = 1e-10
        return int(math.floor(-math.log(r) * self.ml))

    # ---- Layer helpers ---------------------------------------------------

    def _ensure_layer(self, level: int) -> None:
        """Create the per-node dict for *level* if it does not exist."""
        if level not in self.layers:
            self.layers[level] = {}

    def _get_neighbors(self, node_id: str, level: int) -> Set[str]:
        """Return the set of neighbor ids for *node_id* at *level*."""
        if level not in self.layers:
            return set()
        return self.layers[level].get(node_id, set())

    def _add_edge(self, from_id: str, to_id: str, level: int) -> None:
        """Add a one-way edge at *level*."""
        self._ensure_layer(level)
        if from_id not in self.layers[level]:
            self.layers[level][from_id] = set()
        self.layers[level][from_id].add(to_id)

    def _prune_connections(
        self, node_id: str, level: int, max_connections: int
    ) -> None:
        """Prune a node's connections down to *max_connections*.

        Keeps the *max_connections* neighbors that are closest to the
        node according to the current distance metric.  Drops the
        reverse edge for every removed connection to maintain
        bidirectionality.
        """
        if level not in self.layers or node_id not in self.layers[level]:
            return

        neighbors = self.layers[level][node_id]
        if len(neighbors) <= max_connections:
            return

        node_vec = self.vectors[node_id]
        scored: List[Tuple[str, float]] = []
        for n in neighbors:
            if n in self.vectors:
                d = self._distance(node_vec, self.vectors[n])
            else:
                d = float("inf")
            scored.append((n, d))

        scored.sort(key=lambda x: x[1])
        keep = {n for n, _ in scored[:max_connections]}
        remove = neighbors - keep

        # Remove reverse edges for every dropped connection
        for n in remove:
            if level in self.layers and n in self.layers[level]:
                self.layers[level][n].discard(node_id)

        self.layers[level][node_id] = keep

    # ---- Neighbor selection ----------------------------------------------

    def _select_neighbors(
        self,
        candidates: List[Tuple[str, float]],
        M: int,
    ) -> List[str]:
        """Select up to *M* neighbors using heuristic pruning.

        A candidate is kept only if it is closer to the query than to
        every already-selected neighbor.  This avoids dense clusters
        and produces a more navigable graph.
        """
        result: List[str] = []
        discarded: Set[str] = set()

        sorted_candidates = sorted(candidates, key=lambda x: x[1])

        for c_id, c_dist in sorted_candidates:
            if c_id in discarded:
                continue

            keep = True
            for r_id in result:
                if r_id in self.vectors and c_id in self.vectors:
                    r_dist_to_c = self._distance(
                        self.vectors[c_id], self.vectors[r_id]
                    )
                    if r_dist_to_c < c_dist:
                        keep = False
                        break

            if keep:
                result.append(c_id)
                if len(result) >= M:
                    break

        # Fill remaining slots with the closest unselected candidates
        if len(result) < M:
            for c_id, __ in sorted_candidates:
                if c_id not in result and c_id not in discarded:
                    result.append(c_id)
                    if len(result) >= M:
                        break

        return result

    # ---- Search primitives -----------------------------------------------

    def _greedy_search_layer(
        self,
        query: np.ndarray,
        entry_id: str,
        level: int,
    ) -> Tuple[str, float]:
        """Greedy descent on a single layer.

        Starting from *entry_id*, repeatedly move to a neighbor that
        is closer to *query*.  Returns the local minimum and its
        distance.

        Parameters
        ----------
        query : np.ndarray
            Query vector (1D).
        entry_id : str
            ID of the node to start from.
        level : int
            The graph layer to search.

        Returns
        -------
        Tuple[str, float]
            ``(best_node_id, distance_to_query)``.
        """
        current = entry_id
        current_dist = self._distance(query, self.vectors[current])

        while True:
            changed = False
            for neighbor in self._get_neighbors(current, level):
                if neighbor in self.deleted:
                    continue
                if neighbor not in self.vectors:
                    continue
                n_dist = self._distance(query, self.vectors[neighbor])
                if n_dist < current_dist:
                    current = neighbor
                    current_dist = n_dist
                    changed = True
            if not changed:
                break

        return current, current_dist

    def _beam_search_layer(
        self,
        query: np.ndarray,
        entry_ids: List[str],
        ef: int,
        level: int,
    ) -> List[Tuple[str, float]]:
        """Beam search on a single layer.

        Returns up to *ef* nearest neighbors found during exploration.

        Parameters
        ----------
        query : np.ndarray
            Query vector (1D).
        entry_ids : List[str]
            IDs of starting nodes.
        ef : int
            Beam width — explore until *ef* neighbors are collected.
        level : int
            The graph layer to search.

        Returns
        -------
        List[Tuple[str, float]]
            List of ``(node_id, distance)`` tuples sorted by distance.
        """
        visited: Set[str] = set()
        candidates: List[Tuple[float, str]] = []  # min-heap: (dist, id)
        results: List[Tuple[float, str]] = []  # max-heap: (-dist, id)

        for ep in entry_ids:
            if ep in self.deleted or ep not in self.vectors:
                continue
            d = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
            visited.add(ep)

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)

            worst_dist = -results[0][0] if results else float("inf")

            if c_dist > worst_dist:
                break

            for neighbor in self._get_neighbors(c_id, level):
                if neighbor in visited:
                    continue
                if neighbor in self.deleted:
                    continue
                if neighbor not in self.vectors:
                    continue

                visited.add(neighbor)
                n_dist = self._distance(query, self.vectors[neighbor])

                if n_dist < worst_dist or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor))
                    heapq.heappush(results, (-n_dist, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(n_id, -d) for d, n_id in results]

    # ---- Entry-point repair ----------------------------------------------

    def _repair_entry_point(self) -> None:
        """Choose a new entry point as the highest-level non-deleted node."""
        max_level = -1
        new_ep: Optional[str] = None
        for node_id, level in self.node_levels.items():
            if node_id not in self.deleted and node_id in self.vectors:
                if level > max_level:
                    max_level = level
                    new_ep = node_id
        self.entry_point = new_ep

    # ---- Public mutators -------------------------------------------------

    def add(self, vector: np.ndarray, id_: str) -> None:
        """Add or update a vector in the index.

        Parameters
        ----------
        vector : np.ndarray
            1D array of shape ``(dim,)``.
        id_ : str
            Unique identifier for the vector.
        """
        if id_ in self.vectors:
            old_level = self.node_levels[id_]
            for lvl in range(old_level + 1):
                if lvl in self.layers and id_ in self.layers[lvl]:
                    for nb in list(self.layers[lvl][id_]):
                        if lvl in self.layers and nb in self.layers[lvl]:
                            self.layers[lvl][nb].discard(id_)
                    del self.layers[lvl][id_]

            self.vectors[id_] = vector.copy()
            new_level = self._random_level()
            self.node_levels[id_] = new_level
            self.deleted.discard(id_)
            self._insert_node(id_, vector)
            return

        self.deleted.discard(id_)
        self.vectors[id_] = vector.copy()
        level = self._random_level()
        self.node_levels[id_] = level
        self._insert_node(id_, vector)

    def _insert_node(self, id_: str, vector: np.ndarray) -> None:
        """Insert a new node into the HNSW graph."""
        level = self.node_levels[id_]

        if self.entry_point is None:
            self.entry_point = id_
            for lvl in range(level + 1):
                self._ensure_layer(lvl)
                self.layers[lvl][id_] = set()
            return

        if self.entry_point in self.deleted or self.entry_point not in self.vectors:
            self._repair_entry_point()

        curr_ep = self.entry_point
        if curr_ep is None:
            self.entry_point = id_
            for lvl in range(level + 1):
                self._ensure_layer(lvl)
                self.layers[lvl][id_] = set()
            return

        prev_ep_level = self.node_levels[curr_ep]

        # Greedy descent from top of entry point down to *level + 1*
        for lc in range(prev_ep_level, level, -1):
            curr_ep, __ = self._greedy_search_layer(vector, curr_ep, lc)

        # Insert at layers min(level, prev_ep_level) down to 0
        top_insert = min(level, prev_ep_level)
        for lc in range(top_insert, -1, -1):
            results = self._beam_search_layer(
                vector, [curr_ep], self.ef_construction, lc
            )
            max_conn = self.M_max0 if lc == 0 else self.M_max
            selected = self._select_neighbors(results, max_conn)

            self._ensure_layer(lc)
            self.layers[lc][id_] = set(selected)
            for nb in selected:
                self._add_edge(nb, id_, lc)

            for nb in selected:
                self._prune_connections(nb, lc, max_conn)
                # If pruning removed us from nb's side, drop nb from our side
                if id_ not in self.layers[lc].get(nb, set()):
                    self.layers[lc][id_].discard(nb)

            if results and lc > 0:
                curr_ep = results[0][0]

        if level > prev_ep_level:
            self.entry_point = id_

    def search(
        self, query: np.ndarray, k: int = 10, ef: int = 50
    ) -> List[Tuple[str, float]]:
        """Search for the *k* nearest neighbors.

        Parameters
        ----------
        query : np.ndarray
            1D query vector of shape ``(dim,)``.
        k : int, optional
            Number of results to return. Default is 10.
        ef : int, optional
            Beam width for layer-0 search. Larger values improve
            recall at the cost of search time. Must be >= *k*.
            Default is 50.

        Returns
        -------
        List[Tuple[str, float]]
            List of ``(id, score)``, where *score* is a similarity
            score (higher is more similar).  Returns an empty list
            when the index is empty.
        """
        if self.entry_point is None or self.element_count == 0:
            return []

        ep = self.entry_point
        if ep in self.deleted or ep not in self.vectors:
            self._repair_entry_point()
            if self.entry_point is None:
                return []
            ep = self.entry_point

        ep_level = self.node_levels[ep]

        # Greedy descent from top layer down to layer 1
        curr_ep = ep
        for lc in range(ep_level, 0, -1):
            curr_ep, __ = self._greedy_search_layer(query, curr_ep, lc)

        # Beam search at layer 0
        ef_search = max(ef, k)
        results = self._beam_search_layer(query, [curr_ep], ef_search, 0)

        results.sort(key=lambda x: x[1])
        top_k = results[:k]

        # Convert internal distance to similarity score
        scores: List[Tuple[str, float]] = []
        for n_id, dist in top_k:
            if self.metric == "cosine":
                score = 1.0 - dist
            elif self.metric == "euclidean":
                score = -dist
            elif self.metric == "dot_product":
                score = -dist
            else:
                score = -dist
            scores.append((n_id, score))

        return scores

    def delete(self, id_: str) -> None:
        """Mark a vector as deleted (soft delete).

        The node is excluded from subsequent searches but its edges
        remain in the graph as tombstones.

        Parameters
        ----------
        id_ : str
            Identifier of the vector to delete.
        """
        if id_ in self.vectors:
            self.deleted.add(id_)
            if id_ == self.entry_point:
                self._repair_entry_point()

    # ---- Serialization ---------------------------------------------------

    def get_state(self) -> dict:
        """Return a JSON-friendly representation of the index state."""
        serializable_layers: dict = {}
        for level, node_dict in self.layers.items():
            serializable_layers[level] = {}
            for node_id, neighbors in node_dict.items():
                serializable_layers[level][node_id] = list(neighbors)

        return {
            "layers": serializable_layers,
            "node_levels": dict(self.node_levels),
            "entry_point": self.entry_point,
            "deleted": list(self.deleted),
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
        dim: int,
        metric: str,
        M: int,
        ef_construction: int,
    ) -> "HNSWIndex":
        """Restore an index from a state dict previously returned by
        :meth:`get_state`.

        Parameters
        ----------
        state : dict
            State dictionary.
        dim : int
            Dimensionality of the vectors.
        metric : str
            Distance metric.
        M : int
            Max connections per node in higher layers.
        ef_construction : int
            Beam width used during construction.

        Returns
        -------
        HNSWIndex
            Restored index.
        """
        index = cls(dim=dim, metric=metric, M=M, ef_construction=ef_construction)

        index.vectors = {k: np.array(v, dtype=np.float64) for k, v in state["vectors"].items()}
        index.node_levels = state["node_levels"]
        index.entry_point = state["entry_point"]
        index.deleted = set(state.get("deleted", []))

        index.layers = {}
        for level_key, node_dict in state["layers"].items():
            level = int(level_key)
            index.layers[level] = {}
            for node_id, neighbors in node_dict.items():
                index.layers[level][node_id] = set(neighbors)

        return index
