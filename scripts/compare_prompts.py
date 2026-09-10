"""Compare two prompt versions on the same dataset."""

import argparse
import json
from pathlib import Path

import tiktoken

from llmapp.config import Settings
from llmapp.llm.factory import build_client
from llmapp.prompts.registry import PromptRegistry

CATEGORIES = "billing, shipping, technical, other"
ENCODING = tiktoken.get_encoding("o200k_base")


def load_cases(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/triage_tickets.jsonl")
    parser.add_argument("--versions", default="1,2")
    args = parser.parse_args()

    settings = Settings()
    client = build_client(settings)
    registry = PromptRegistry(Path("prompts"))
    cases = load_cases(Path(args.dataset))
    live = settings.provider != "fake"

    print(f"{'version':>8} {'prompt_id':>22} "
          f"{'tokens':>7} {'accuracy':>9}")
    for raw in args.versions.split(","):
        version = int(raw)
        template = registry.get("triage", version)
        tokens = 0
        correct = 0
        prompt_id = ""
        for case in cases:
            available = {
                "ticket": case["ticket"],
                "categories": CATEGORIES,
            }
            prompt = template.render(
                **{
                    k: v
                    for k, v in available.items()
                    if k in template.variables
                }
            )
            prompt_id = prompt.prompt_id
            tokens += sum(
                len(ENCODING.encode(m.content))
                for m in prompt.messages
            )
            if live:
                reply = client.complete(prompt.messages).text
                if reply.strip().lower() == case["expected"]:
                    correct += 1
        score = (
            f"{correct / len(cases):.0%}" if live else "n/a"
        )
        print(f"{version:>8} {prompt_id:>22} "
              f"{tokens:>7} {score:>9}")

    if not live:
        print("\nAccuracy needs a real provider. Token counts "
              "above are exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
