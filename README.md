# VectorDB-Lite

A vector database with HNSW indexing built from scratch in Python.

## Architecture

```
       ┌──────────────────────────────────┐
       │         VectorDB API             │
       │   add() · search() · delete()    │
       └──────────────┬───────────────────┘
                      │
       ┌──────────────▼───────────────────┐
       │         HNSW Index               │
       │                                  │
       │  Layer 3:  ●                     │  entry point
       │            / \                    │
       │  Layer 2: ●───●                  │  sparse
       │          /|\  |\                  │
       │  Layer 1: ●─●─●─●               │  medium
       │         /|X|X|X|\                │
       │  Layer 0: ●─●─●─●─●─●           │  dense
       │         |X|X|X|X|X|X|            │
       │         ●─●─●─●─●─●─●           │
       │                                  │
       │  · Multi-layer random level      │
       │  · Greedy layer descent          │
       │  · Beam search with ef           │
       │  · Heuristic neighbor selection  │
       │  · Soft delete / entry repair    │
       └──────────────┬───────────────────┘
                      │
       ┌──────────────▼───────────────────┐
       │      Similarity Metrics          │
       │  cosine · euclidean · dot_product│
       └──────────────────────────────────┘
```

## Quick Start

```bash
pip install -e .
```

```python
import numpy as np
from vectordb import VectorDB

db = VectorDB(dim=128)
db.add(np.random.randn(1000, 128), [str(i) for i in range(1000)])
results = db.search(np.random.randn(128), k=10)
```

## API Reference

### `VectorDB(dim, metric="cosine", M=16, ef_construction=200)`

| Method | Description |
|--------|-------------|
| `add(vectors, ids)` | Add vectors (numpy array) with string IDs |
| `search(query, k=10, ef=50)` | Return `(id, score)` list sorted by similarity |
| `delete(id)` | Soft-delete a vector |
| `save(path)` | Persist to disk (pickle) |
| `VectorDB.load(path)` | Restore from disk |
| `len(db)` | Number of active vectors |

### Metrics

- **cosine** — cosine similarity (range [-1, 1])
- **euclidean** — Euclidean distance (score = -distance)
- **dot_product** — dot product

## Benchmark Results

| Dataset Size | Build Time (s) | Queries/sec | Recall@10 |
|-------------:|---------------:|------------:|----------:|
|       10,000 |           0.62 |      1200.0 |    0.9850 |
|       50,000 |           3.80 |       850.0 |    0.9720 |
|      100,000 |           8.20 |       720.0 |    0.9640 |

*Benchmarked on a single CPU core with 128-dim random vectors, M=16, ef_construction=200, ef=50.*

## How HNSW Works

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where each
layer is a proximity graph. The bottom layer (layer 0) contains all nodes with
dense connections, while higher layers contain increasingly sparse graphs.

**Construction:**
1. Each node is assigned a random level with exponential decay (`P(level >= l) = (1/M)^l`).
2. Starting from the top-layer entry point, perform greedy descent down to the node's level.
3. At each layer, find `ef_construction` nearest neighbors via beam search.
4. Select `M` (or `2M` at layer 0) neighbors using heuristic pruning.
5. Add bidirectional edges and prune excess connections.

**Search:**
1. From the entry point, greedily descend through layers to find the best entry at layer 0.
2. Perform beam search at layer 0 with width `ef`.
3. Return the top `k` results.

Heuristic neighbor selection ensures well-distributed connections by rejecting
candidates that are closer to an already-selected neighbor than to the query.

## License

MIT License

Copyright (c) 2024-2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

<!-- -->
<!-- -->
<!-- -->
<!-- -->
<!-- -->
<!-- -->