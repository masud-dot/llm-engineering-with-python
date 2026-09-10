"""Model output is untrusted input to whatever consumes it."""

import html
import re
from dataclasses import dataclass
from urllib.parse import urlparse

MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
SAFE_SCHEMES = frozenset({"https"})


class UnsafeOutput(ValueError):
    """Generated text failed a check before being used."""


def escape_for_html(text: str) -> str:
    """Never render generated text as markup."""
    return html.escape(text, quote=True)


@dataclass(frozen=True)
class UrlPolicy:
    """Which links a generated answer may contain."""

    allowed_hosts: frozenset[str]
    schemes: frozenset[str] = SAFE_SCHEMES

    def permits(self, url: str) -> bool:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in self.schemes:
            return False
        host = (parsed.hostname or "").lower()
        return any(
            host == allowed or host.endswith("." + allowed)
            for allowed in self.allowed_hosts
        )


def find_links(text: str) -> list[str]:
    """Every URL a markdown answer would render."""
    return [
        target.strip()
        for _, target in MARKDOWN_LINK.findall(text)
    ]


def check_links(text: str, policy: UrlPolicy) -> list[str]:
    """Links the policy would not permit."""
    return [
        url for url in find_links(text) if not policy.permits(url)
    ]


def strip_disallowed_links(
    text: str, policy: UrlPolicy
) -> str:
    """Keep the label, drop the destination."""

    def replace(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2).strip()
        if policy.permits(target):
            return match.group(0)
        return label

    return MARKDOWN_LINK.sub(replace, text)
