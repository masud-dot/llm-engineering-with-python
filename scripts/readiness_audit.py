"""The production readiness checklist, as a runnable check."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from llmapp.config import Settings


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def run_checks() -> list[Check]:
    checks: list[Check] = []

    try:
        settings = Settings()
        checks.append(Check("configuration loads", True))
    except Exception as exc:
        return [Check("configuration loads", False, str(exc))]

    secret = settings.jwt_secret
    checks.append(
        Check(
            "secrets are not defaults",
            secret is not None
            and len(secret.get_secret_value()) >= 32,
            "LLMAPP_JWT_SECRET must be 32+ bytes",
        )
    )
    checks.append(
        Check(
            "a spend ceiling is set",
            settings.daily_cost_limit_usd > 0,
            "LLMAPP_DAILY_COST_LIMIT_USD is zero",
        )
    )
    checks.append(
        Check(
            "a request timeout is set",
            settings.request_timeout_s > 0,
        )
    )
    checks.append(
        Check(
            "the fake provider is not in production",
            not (
                settings.environment == "production"
                and settings.provider == "fake"
            ),
        )
    )
    checks.append(
        Check(
            "prompts are present",
            Path("prompts").is_dir()
            and any(Path("prompts").glob("*.toml")),
        )
    )
    checks.append(
        Check(
            "an evaluation dataset exists",
            Path("evals/triage.jsonl").exists(),
            "no dataset means no quality gate",
        )
    )
    checks.append(
        Check(
            "an adversarial suite exists",
            Path("evals/adversarial.jsonl").exists(),
        )
    )
    checks.append(
        Check(
            "no .env is deployed",
            not Path(".env").exists()
            or settings.environment != "production",
            "production reads the platform secret store",
        )
    )
    checks.append(
        Check(
            "the tokenizer cache is warm",
            bool(os.environ.get("TIKTOKEN_CACHE_DIR")),
            "the first request will need the network",
        )
    )
    return checks


def main() -> int:
    checks = run_checks()
    failed = [c for c in checks if not c.passed]
    for check in checks:
        mark = "ok" if check.passed else "XX"
        line = f"[{mark}] {check.name}"
        if not check.passed and check.detail:
            line += f" — {check.detail}"
        print(line)
    print()
    print(
        f"{len(checks) - len(failed)}/{len(checks)} checks passed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
