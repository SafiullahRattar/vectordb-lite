"""Public API for VectorDB-Lite.

Provides a simple, scikit-learn–style interface backed by the
:class:`HNSWIndex` graph and pickle-based persistence.
"""

from typing import List, Optional, Tuple

import numpy as np

from vectordb.index import HNSWIndex
from vectordb.persistence import save_index, load_index


class VectorDB:
    """Approximate nearest-neighbor vector database.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors.
    metric : str, optional
        Distance metric. One of ``"cosine"``, ``"euclidean"``, or
        ``"dot_product"``.  Default is ``"cosine"``.
    M : int, optional
        Max connections per node in higher HNSW layers.  Default is 16.
    ef_construction : int, optional
        Beam width used during index construction.  Default is 200.

    Examples
    --------
    >>> import numpy as np
    >>> db = VectorDB(dim=128)
    >>> db.add(np.random.randn(10, 128), [str(i) for i in range(10)])
    >>> results = db.search(np.random.randn(128), k=3)
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
    ) -> None:
        self._index = HNSWIndex(
            dim=dim, metric=metric, M=M, ef_construction=ef_construction
        )

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the database.

        Parameters
        ----------
        vectors : np.ndarray
            Array of shape ``(n, dim)`` or ``(dim,)``.  If 1D it is
            treated as a single vector.
        ids : List[str]
            String identifiers. Must match the number of vectors.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 1D or 2D")
        if vectors.shape[1] != self._index.dim:
            raise ValueError(
                f"Expected vectors of dim {self._index.dim}, "
                f"got {vectors.shape[1]}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of ids ({len(ids)}) does not match "
                f"number of vectors ({vectors.shape[0]})"
            )
        for i, id_ in enumerate(ids):
            self._index.add(vectors[i], id_)

    def search(
        self, query: np.ndarray, k: int = 10, ef: int = 50
    ) -> List[Tuple[str, float]]:
        """Search for *k* nearest neighbors.

        Parameters
        ----------
        query : np.ndarray
            1D query vector of shape ``(dim,)``.
        k : int, optional
            Number of results.  Default is 10.
        ef : int, optional
            Beam width for the layer-0 search.  Default is 50.

        Returns
        -------
        List[Tuple[str, float]]
            List of ``(id, score)`` tuples sorted by descending
            similarity.

        Raises
        ------
        ValueError
            If *query* has the wrong dimensionality.
        """
        query = np.asarray(query, dtype=np.float64)
        if query.ndim != 1:
            raise ValueError("query must be a 1D array")
        if query.shape[0] != self._index.dim:
            raise ValueError(
                f"Expected query of dim {self._index.dim}, "
                f"got {query.shape[0]}"
            )
        return self._index.search(query, k=k, ef=ef)

    def delete(self, id_: str) -> None:
        """Soft-delete a vector by its identifier.

        Parameters
        ----------
        id_ : str
            Identifier of the vector to delete.
        """
        self._index.delete(id_)

    def save(self, path: str) -> None:
        """Persist the database to disk.

        Parameters
        ----------
        path : str
            File path for the serialized index.
        """
        save_index(self._index, path)

    @classmethod
    def load(cls, path: str) -> Optional["VectorDB"]:
        """Restore a database from disk.

        Parameters
        ----------
        path : str
            File path to read from.

        Returns
        -------
        VectorDB or None
            The restored database, or ``None`` if the file does not
            exist or is corrupt.
        """
        index = load_index(path)
        if index is None:
            return None
        db = cls.__new__(cls)
        db._index = index  # type: ignore[attr-defined]
        return db

    def __len__(self) -> int:
        """Return the number of active (non-deleted) vectors."""
        return self._index.element_count
