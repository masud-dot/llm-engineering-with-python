"""Measure recall and latency for three FAISS index types.

Pass 'clustered' (default) or 'uniform' to change the data.
Real embeddings have cluster structure; uniform random
vectors do not, and the difference is the point.
"""

import sys
import time

try:
    import faiss
except ImportError:  # optional measurement extra
    raise SystemExit(
        "index_tradeoff.py needs FAISS: pip install -e '.[bench]'"
    )
import numpy as np

N, DIM, QUERIES, K, CLUSTERS = 50_000, 256, 200, 10, 300


def build_data(kind: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    if kind == "uniform":
        data = rng.normal(size=(N, DIM))
        queries = rng.normal(size=(QUERIES, DIM))
    else:
        centres = rng.normal(size=(CLUSTERS, DIM))
        pick = rng.integers(0, CLUSTERS, size=N)
        data = centres[pick] + 0.35 * rng.normal(size=(N, DIM))
        qpick = rng.integers(0, CLUSTERS, size=QUERIES)
        queries = centres[qpick] + 0.35 * rng.normal(
            size=(QUERIES, DIM)
        )
    data32 = data.astype("float32")
    queries32 = queries.astype("float32")
    faiss.normalize_L2(data32)
    faiss.normalize_L2(queries32)
    return data32, queries32


def timed_search(
    index: faiss.Index, queries: np.ndarray
) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    _, ids = index.search(queries, K)
    elapsed = (time.perf_counter() - start) / len(queries) * 1000
    return elapsed, ids


def main() -> int:
    kind = sys.argv[1] if len(sys.argv) > 1 else "clustered"
    data, queries = build_data(kind)

    flat = faiss.IndexFlatIP(DIM)
    start = time.perf_counter()
    flat.add(data)
    flat_build = time.perf_counter() - start
    flat_ms, truth = timed_search(flat, queries)

    def recall(ids: np.ndarray) -> float:
        shared = sum(
            len(set(a) & set(b))
            for a, b in zip(truth, ids, strict=True)
        )
        return shared / (len(queries) * K)

    print(f"data: {kind}, {N} vectors, {DIM} dimensions")
    print(f"{'index':<22}{'build s':>9}{'ms/query':>10}"
          f"{'recall@10':>11}")
    print(f"{'flat (exact)':<22}{flat_build:>9.2f}"
          f"{flat_ms:>10.3f}{1.00:>11.2f}")

    quantizer = faiss.IndexFlatIP(DIM)
    ivf = faiss.IndexIVFFlat(
        quantizer, DIM, 256, faiss.METRIC_INNER_PRODUCT
    )
    start = time.perf_counter()
    ivf.train(data)
    ivf.add(data)
    ivf_build = time.perf_counter() - start
    for nprobe in (1, 8, 32):
        ivf.nprobe = nprobe
        ms, ids = timed_search(ivf, queries)
        print(f"{'ivf nprobe=' + str(nprobe):<22}"
              f"{ivf_build:>9.2f}{ms:>10.3f}{recall(ids):>11.2f}")

    hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    start = time.perf_counter()
    hnsw.add(data)
    hnsw_build = time.perf_counter() - start
    for ef_search in (16, 64, 256):
        hnsw.hnsw.efSearch = ef_search
        ms, ids = timed_search(hnsw, queries)
        print(f"{'hnsw efSearch=' + str(ef_search):<22}"
              f"{hnsw_build:>9.2f}{ms:>10.3f}{recall(ids):>11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
