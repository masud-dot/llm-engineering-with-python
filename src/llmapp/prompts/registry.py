"""Prompts are versioned files, not string literals."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from string import Template

from llmapp.llm.base import Message


class PromptError(Exception):
    """The prompt is missing, malformed, or misused."""


@dataclass(frozen=True)
class RenderedPrompt:
    """One prompt, ready to send, with its identity."""

    name: str
    version: int
    fingerprint: str
    messages: list[Message]

    @property
    def prompt_id(self) -> str:
        return f"{self.name}.v{self.version}.{self.fingerprint}"


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt file loaded from disk."""

    name: str
    version: int
    instruction: str
    user_block: str
    variables: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        """Content hash, so a silent edit cannot hide."""
        blob = (self.instruction + self.user_block).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def render(self, **values: str) -> RenderedPrompt:
        missing = set(self.variables) - set(values)
        if missing:
            raise PromptError(
                f"{self.name} v{self.version} needs "
                f"{sorted(missing)}"
            )
        extra = set(values) - set(self.variables)
        if extra:
            raise PromptError(
                f"{self.name} v{self.version} does not declare "
                f"{sorted(extra)}"
            )
        # Instructions may not carry user data.
        instruction = Template(self.instruction).substitute(
            {k: v for k, v in values.items() if k != "ticket"}
        ).strip()
        body = Template(self.user_block).substitute(values).strip()
        return RenderedPrompt(
            name=self.name,
            version=self.version,
            fingerprint=self.fingerprint,
            messages=[
                Message(role="system", content=instruction),
                Message(role="user", content=body),
            ],
        )


class PromptRegistry:
    """Load prompt files and hand out templates."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[tuple[str, int], PromptTemplate] = {}

    def get(self, name: str, version: int) -> PromptTemplate:
        key = (name, version)
        if key in self._cache:
            return self._cache[key]
        path = self._root / f"{name}.v{version}.toml"
        if not path.exists():
            raise PromptError(f"no prompt file at {path}")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if data.get("name") != name:
            # A flat namespace lets two subsystems claim one
            # name, and the second loaded would silently win.
            raise PromptError(
                f"{path.name} declares name="
                f"{data.get('name')!r}, expected {name!r}"
            )
        template = PromptTemplate(
            name=data["name"],
            version=int(data["version"]),
            instruction=data["instruction"],
            user_block=data["user_block"],
            variables=tuple(data.get("variables", [])),
        )
        self._cache[key] = template
        return template

    def versions(self, name: str) -> list[int]:
        found = self._root.glob(f"{name}.v*.toml")
        return sorted(
            int(p.name.split(".v")[1].split(".")[0]) for p in found
        )
