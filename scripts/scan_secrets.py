"""Fail if anything that looks like a credential is tracked."""

import pathlib
import re
import sys

PATTERNS = {
    "openai key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "aws key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic assignment": re.compile(
        r"(?i)(api_key|secret|password)\s*=\s*['\"][^'\"]{12,}"
    ),
}
SKIP = {".git", ".venv", "__pycache__", ".mypy_cache"}
ALLOW = {".env.example"}


def scan(root: pathlib.Path) -> int:
    findings = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP for part in path.parts):
            continue
        if path.name in ALLOW:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                print(f"{path}:{line}: possible {label}")
                findings += 1
    return findings


if __name__ == "__main__":
    count = scan(pathlib.Path("."))
    print(f"{count} finding(s)")
    sys.exit(1 if count else 0)
