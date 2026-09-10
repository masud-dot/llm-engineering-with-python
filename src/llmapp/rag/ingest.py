"""Turn files into chunks that carry their provenance."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from llmapp.retrieval.chunking import Chunk, chunk_text

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
MULTI_SPACE = re.compile(r"[ \t]+")
MULTI_BLANK = re.compile(r"\n{3,}")
SUPPORTED = (".md", ".txt", ".html", ".htm", ".pdf")


@dataclass(frozen=True)
class Section:
    """A titled part of a document."""

    heading: str
    body: str


class _TextExtractor(HTMLParser):
    """Keep text, drop markup, respect block boundaries."""

    BLOCK = {"p", "div", "li", "br", "tr", "h1", "h2", "h3"}
    SKIP = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipping = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in self.SKIP:
            self._skipping = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skipping = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def normalize(text: str) -> str:
    """Undo the damage that layout does to sentences."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_BLANK.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n"))


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_html(path: Path) -> str:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.text()


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        pages.append(f"## Page {number}\n{page.extract_text()}")
    return "\n\n".join(pages)


def load(path: Path) -> str:
    """Read any supported file into normalized text."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported file type: {suffix}")
    if suffix == ".pdf":
        raw = read_pdf(path)
    elif suffix in (".html", ".htm"):
        raw = read_html(path)
    else:
        raw = read_markdown(path)
    return normalize(raw)


def split_sections(text: str) -> list[Section]:
    """Split on headings so chunks never cross a topic."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [Section(heading="", body=text.strip())]
    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(heading="", body=preamble))
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        body = text[match.end() : end].strip()
        if body:
            sections.append(
                Section(heading=match.group(2).strip(), body=body)
            )
    return sections


def ingest_file(
    path: Path,
    *,
    max_words: int = 120,
    overlap_words: int = 20,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """One file in, provenance-carrying chunks out."""
    text = load(path)
    chunks: list[Chunk] = []
    for section in split_sections(text):
        extra = dict(metadata or {})
        if section.heading:
            extra["heading"] = section.heading
        for chunk in chunk_text(
            section.body,
            path.name,
            max_words=max_words,
            overlap_words=overlap_words,
            metadata=extra,
        ):
            chunks.append(
                Chunk(
                    text=chunk.text,
                    source=path.name,
                    ordinal=len(chunks),
                    metadata=chunk.metadata,
                )
            )
    return chunks


def ingest_directory(
    root: Path, **kwargs: object
) -> dict[str, list[Chunk]]:
    """Every supported file under a directory, by file name."""
    found: dict[str, list[Chunk]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            found[path.name] = ingest_file(
                path, **kwargs  # type: ignore[arg-type]
            )
    return found
