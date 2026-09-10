"""Count tokens and price a request before sending it.

Chapter 2. Tokens are the unit of cost, of latency, and of the
context limit, so counting them is the first measurement worth
being able to take.
"""

import argparse
import sys
from pathlib import Path

import tiktoken

ENCODING = "o200k_base"


def count(text: str, encoding_name: str = ENCODING) -> int:
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def density(text: str, encoding_name: str = ENCODING) -> float:
    """Characters per token. Varies sharply by script."""
    tokens = count(text, encoding_name)
    return len(text) / tokens if tokens else 0.0


def estimate_usd(
    input_tokens: int,
    output_tokens: int,
    input_per_mtok: float,
    output_per_mtok: float,
) -> float:
    return (
        input_tokens * input_per_mtok
        + output_tokens * output_per_mtok
    ) / 1_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", help="file to read; omit to read stdin"
    )
    parser.add_argument("--encoding", default=ENCODING)
    parser.add_argument("--output-tokens", type=int, default=0)
    parser.add_argument("--input-per-mtok", type=float, default=0.0)
    parser.add_argument("--output-per-mtok", type=float, default=0.0)
    args = parser.parse_args()

    text = (
        Path(args.path).read_text(encoding="utf-8")
        if args.path
        else sys.stdin.read()
    )
    tokens = count(text, args.encoding)
    print(f"characters      {len(text):>10,}")
    print(f"tokens          {tokens:>10,}")
    print(f"chars per token {density(text, args.encoding):>10.2f}")

    if args.input_per_mtok or args.output_per_mtok:
        cost = estimate_usd(
            tokens,
            args.output_tokens,
            args.input_per_mtok,
            args.output_per_mtok,
        )
        print(f"estimated cost  {cost:>10.6f} USD")
        print(f"per 50k calls   {cost * 50_000:>10.2f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
