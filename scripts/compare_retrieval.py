"""Keyword search and semantic search on the same queries."""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from llmapp.config import Settings
from llmapp.retrieval.metrics import evaluate

K = 3


def tokenize(text: str) -> list[str]:
    return [w.strip(".,?!").lower() for w in text.split()]


def main() -> int:
    corpus: dict[str, str] = json.loads(
        Path("evals/retrieval_corpus.json").read_text()
    )
    gold: dict[str, list[str]] = json.loads(
        Path("evals/retrieval_queries.json").read_text()
    )
    names = list(corpus)
    bm25 = BM25Okapi([tokenize(corpus[n]) for n in names])

    keyword: dict[str, list[str]] = {}
    for query in gold:
        scores = bm25.get_scores(tokenize(query))
        # A zero BM25 score is no signal at all. Counting such a
        # document as retrieved inflates recall; ties at zero sort
        # by position in the corpus, not by relevance.
        ranked = sorted(
            [i for i in range(len(names)) if scores[i] > 0],
            key=lambda i: -scores[i]
        )
        keyword[query] = [names[i] for i in ranked[:K]]

    print(f"keyword (BM25):  {evaluate(keyword, gold, k=K)}")

    settings = Settings()
    if settings.provider == "fake":
        print("semantic:        needs a real embedding model")
    print()
    print(f"{'query':<44} {'bm25 top hit':<22} {'ok':>3}")
    for query, relevant in gold.items():
        top = keyword[query][0]
        mark = "yes" if top in relevant else "no"
        print(f"{query[:43]:<44} {top:<22} {mark:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
