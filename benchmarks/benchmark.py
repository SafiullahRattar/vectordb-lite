"""Benchmark suite for vectordb-lite.

Measures build time, queries per second, and recall@10 against brute-force
for dataset sizes of 10K, 50K, and 100K 128-dimensional vectors.

Usage::

    python benchmarks/benchmark.py              # all sizes
    python benchmarks/benchmark.py --size 10000 # 10K only
"""

import argparse
import time

import numpy as np

from vectordb import VectorDB

DIM = 128
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50
K = 10


def brute_force_search(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> np.ndarray:
    """Exact k-NN via brute force."""
    from vectordb.similarity import cosine_similarity, euclidean_distance, dot_product

    if metric == "cosine":
        scores = np.array([cosine_similarity(vectors, query)])
    elif metric == "euclidean":
        scores = np.array([euclidean_distance(vectors, query)])
    elif metric == "dot_product":
        scores = np.array([dot_product(vectors, query)])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if metric == "euclidean":
        top_k = np.argsort(scores[0])[:k]
    else:
        top_k = np.argsort(scores[0])[::-1][:k]

    return top_k


def compute_recall(
    hnsw_ids: list,
    brute_ids: np.ndarray,
    id_to_idx: dict,
) -> float:
    """Compute recall@K between HNSW results and brute-force results."""
    hnsw_indices = {id_to_idx[id_] for id_ in hnsw_ids if id_ in id_to_idx}
    brute_set = set(brute_ids)
    if len(brute_set) == 0:
        return 0.0
    return len(hnsw_indices & brute_set) / len(brute_set)


def run_benchmark(n_vectors: int, num_queries: int = 100) -> dict:
    """Run benchmark for a given number of vectors."""
    rng = np.random.RandomState(42)
    print(f"\n  Generating {n_vectors:,} {DIM}-dim vectors ...")

    vectors = rng.randn(n_vectors, DIM).astype(np.float64)
    ids = [str(i) for i in range(n_vectors)]
    queries = rng.randn(num_queries, DIM).astype(np.float64)

    # Build index
    db = VectorDB(dim=DIM, metric="cosine", M=M, ef_construction=EF_CONSTRUCTION)

    t0 = time.perf_counter()
    db.add(vectors, ids)
    build_time = time.perf_counter() - t0

    print(f"  Build: {build_time:.2f}s")

    # Warm-up
    for q in queries[:10]:
        db.search(q, k=K, ef=EF_SEARCH)

    # Search benchmark
    t0 = time.perf_counter()
    all_results = []
    for q in queries:
        results = db.search(q, k=K, ef=EF_SEARCH)
        all_results.append([r[0] for r in results])
    search_time = time.perf_counter() - t0
    qps = num_queries / search_time

    print(f"  Search: {search_time:.2f}s ({qps:.1f} queries/sec)")

    # Recall
    id_to_idx = {str(i): i for i in range(n_vectors)}
    total_recall = 0.0
    for i, q in enumerate(queries):
        brute_ids = brute_force_search(vectors, q, K, metric="cosine")
        total_recall += compute_recall(all_results[i], brute_ids, id_to_idx)
    recall = total_recall / num_queries

    print(f"  Recall@{K}: {recall:.4f}")

    return {
        "n_vectors": n_vectors,
        "build_time_s": build_time,
        "queries_per_sec": qps,
        f"recall@{K}": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="vectordb-lite benchmarks")
    parser.add_argument(
        "--size", type=int, default=None, help="Dataset size (default: all)"
    )
    args = parser.parse_args()

    sizes = [10_000, 50_000, 100_000]
    if args.size is not None:
        sizes = [s for s in sizes if s == args.size]
        if not sizes:
            print(f"Invalid --size {args.size}. Choose from 10000, 50000, 100000.")
            return

    results = []
    for size in sizes:
        results.append(run_benchmark(size))

    # Print markdown table
    print("\n## Benchmark Results\n")
    print("| Dataset Size | Build Time (s) | Queries/sec | Recall@10 |")
    print("|-------------:|---------------:|------------:|----------:|")
    for r in results:
        n = r["n_vectors"]
        b = r["build_time_s"]
        q = r["queries_per_sec"]
        rec = r["recall@10"]
        print(f"| {n:>12,} | {b:>14.2f} | {q:>11.1f} | {rec:>9.4f} |")
    print()


if __name__ == "__main__":
    main()
