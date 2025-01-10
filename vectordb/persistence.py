"""Persistence layer for the HNSW index.

Uses Python's pickle module to save and load index state to disk
with version checking.
"""

import pickle
import os
from datetime import datetime, timezone
from typing import Optional

from vectordb.index import HNSWIndex

CURRENT_VERSION = 1


def save_index(index: HNSWIndex, path: str) -> None:
    """Serialize an :class:`HNSWIndex` to disk.

    Parameters
    ----------
    index : HNSWIndex
        The index to persist.
    path : str
        File path to write the serialized index to.
    """
    metadata = {
        "version": CURRENT_VERSION,
        "dim": index.dim,
        "element_count": index.element_count,
        "metric": index.metric,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    state = index.get_state()

    payload = {"metadata": metadata, "state": state}

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str) -> Optional[HNSWIndex]:
    """Deserialize an :class:`HNSWIndex` from disk.

    Parameters
    ----------
    path : str
        File path to read the serialized index from.

    Returns
    -------
    HNSWIndex or None
        The restored index, or ``None`` if the file does not exist
        or is corrupt.

    Raises
    ------
    RuntimeError
        If the file format version is unsupported.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, Exception):
        return None

    if not isinstance(payload, dict) or "metadata" not in payload or "state" not in payload:
        return None

    metadata = payload["metadata"]
    if metadata.get("version", 0) > CURRENT_VERSION:
        raise RuntimeError(
            f"Unsupported index version {metadata['version']} "
            f"(current: {CURRENT_VERSION})"
        )

    return HNSWIndex.from_state(
        state=payload["state"],
        dim=metadata["dim"],
        metric=metadata["metric"],
        M=metadata.get("M", 16),
        ef_construction=metadata.get("ef_construction", 200),
    )
