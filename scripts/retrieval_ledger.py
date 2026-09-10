"""An improvement ledger: gain per unit of cost and latency.

Every technique is measured against the same query set, one
change at a time. Columns the environment cannot compute are
printed as n/a rather than estimated.
"""

import json
import time
from pathlib import Path

from llmapp.config import Settings
from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.fusion import reciprocal_rank_fusion
from llmapp.retrieval.lexical import BM25Index
from llmapp.retrieval.metrics import evaluate

K = 3


def load_corpus() -> list[Chunk]:
    raw = json.loads(
        Path("evals/retrieval_corpus.json").read_text()
    )
    return [
        Chunk(text=body, source=name, ordinal=0)
        for name, body in raw.items()
    ]


def run(
    index: BM25Index,
    gold: dict[str, list[str]],
    rewrites: dict[str, list[str]] | None,
) -> tuple[dict[str, list[str]], float, float]:
    results: dict[str, list[str]] = {}
    calls = 0.0
    start = time.perf_counter()
    for question in gold:
        variants = [question]
        if rewrites:
            variants += rewrites.get(question, [])
            calls += 1
        rankings = [
            index.search(variant, k=K * 2) for variant in variants
        ]
        fused = reciprocal_rank_fusion(rankings, k=K)
        results[question] = [h.chunk.source for h in fused]
    elapsed = (time.perf_counter() - start) / len(gold) * 1000
    return results, elapsed, calls / len(gold)


def main() -> int:
    corpus = load_corpus()
    gold = json.loads(
        Path("evals/retrieval_queries.json").read_text()
    )
    rewrites = json.loads(
        Path("evals/retrieval_rewrites.json").read_text()
    )
    index = BM25Index()
    index.add(corpus)

    print(f"{'configuration':<26}{'recall@3':>9}{'mrr':>7}"
          f"{'ms/query':>10}{'calls':>7}")
    for label, expansion in (
        ("lexical only", None),
        ("lexical + multi-query", rewrites),
    ):
        results, ms, calls = run(index, gold, expansion)
        score = evaluate(results, gold, k=K)
        print(f"{label:<26}{score.recall_at_k:>9.2f}"
              f"{score.mrr:>7.2f}{ms:>10.3f}{calls:>7.1f}")

    if Settings().provider == "fake":
        for label in ("semantic only", "hybrid (rrf)",
                      "hybrid + rerank"):
            print(f"{label:<26}{'n/a':>9}{'n/a':>7}"
                  f"{'n/a':>10}{'n/a':>7}")
        print("\nRows marked n/a need an embedding model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
