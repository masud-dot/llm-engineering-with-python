"""Run the evaluation and adversarial suites as a gate."""

import argparse
import json
import sys
from pathlib import Path

# evalsetup.py lives at the repository root, beside pyproject.toml.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmapp.eval.adversarial import (
    AdversarialCase,
    Expectation,
    run_suite,
)
from llmapp.eval.regression import (
    compare_to_baseline,
    load_baseline,
    save_baseline,
    snapshot,
)
from llmapp.eval.runner import evaluate_dataset

MIN_PASS_RATE = 0.90
MIN_HELD_RATE = 1.00


def load_adversarial(path: Path) -> list[AdversarialCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            AdversarialCase(
                id=raw["id"],
                question=raw["question"],
                expect=Expectation(raw["expect"]),
                forbidden=tuple(raw.get("forbidden", [])),
                technique=raw.get("technique", ""),
            )
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="")
    parser.add_argument("--record", default="")
    parser.add_argument("--adversarial", action="store_true")
    args = parser.parse_args()

    from evalsetup import build_dataset, build_runner
    from llmapp.config import Settings

    if Settings().provider == "fake":
        print(
            "The quality gate scores real model output, so it needs "
            "a real provider.\n"
            "Set LLMAPP_PROVIDER, LLMAPP_MODEL and LLMAPP_API_KEY, "
            "then re-run.\n"
            "Everything else in this repository runs without a key: "
            "pytest -m 'not local_server and not live'"
        )
        return 2

    dataset = build_dataset()
    runner = build_runner()

    if args.adversarial:
        cases = load_adversarial(Path("evals/adversarial.jsonl"))
        hostile = run_suite(cases, runner)
        print(hostile.summary())
        for breach in hostile.breaches():
            print(f"  BREACH {breach.case_id}: {breach.reason}")
        return 0 if hostile.held_rate >= MIN_HELD_RATE else 1

    report = evaluate_dataset(dataset, runner)
    print(report.summary())

    if args.record:
        path = save_baseline(snapshot(report, args.record))
        print(f"baseline written to {path}")
        return 0

    if report.pass_rate < MIN_PASS_RATE:
        print(
            f"BLOCK: pass rate {report.pass_rate:.0%} below "
            f"{MIN_PASS_RATE:.0%}"
        )
        return 1

    if args.baseline:
        change = compare_to_baseline(
            load_baseline(args.baseline), report
        )
        if not change.comparable:
            print(f"BLOCK: {change.note}")
            return 1
        for case_id in change.fixed:
            print(f"  fixed      {case_id}")
        for case_id in change.broken:
            print(f"  REGRESSED  {case_id}")
        if change.is_regression:
            print("BLOCK: cases that passed now fail")
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
