from pathlib import Path

import pytest

from llmapp.rag.ingest import (
    ingest_directory,
    ingest_file,
    load,
    normalize,
    split_sections,
)

CORPUS = Path("tests/corpus")


def test_markdown_splits_on_headings() -> None:
    sections = split_sections(load(CORPUS / "refunds.md"))
    assert [s.heading for s in sections] == [
        "Refund Policy",
        "Damaged Goods",
    ]


def test_html_markup_and_scripts_are_dropped() -> None:
    text = load(CORPUS / "shipping.html")
    assert "console.log" not in text
    assert "color: red" not in text
    assert "refunded separately" in text


def test_pdf_text_is_extracted() -> None:
    text = load(CORPUS / "escalation.pdf")
    assert "two working days" in text


def test_hyphenation_across_a_line_break_is_repaired() -> None:
    assert normalize("senior engi-\nneer") == "senior engineer"


def test_pdf_hyphenation_survives_extraction() -> None:
    text = load(CORPUS / "escalation.pdf")
    assert "senior engineer" in text
    assert "engi-" not in text


def test_unsupported_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load(Path("notes.docx"))


def test_chunks_carry_source_ordinal_and_heading() -> None:
    chunks = ingest_file(CORPUS / "refunds.md", max_words=40)
    assert all(c.source == "refunds.md" for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata["heading"] == "Refund Policy"


def test_chunks_never_cross_a_heading() -> None:
    chunks = ingest_file(CORPUS / "refunds.md", max_words=200)
    for chunk in chunks:
        head = chunk.metadata["heading"]
        if head == "Refund Policy":
            assert "photograph" not in chunk.text
        if head == "Damaged Goods":
            assert "fourteen days" not in chunk.text


def test_extra_metadata_reaches_every_chunk() -> None:
    chunks = ingest_file(
        CORPUS / "refunds.md",
        max_words=40,
        metadata={"team": "support"},
    )
    assert all(c.metadata["team"] == "support" for c in chunks)


def test_directory_ingest_finds_every_supported_file() -> None:
    found = ingest_directory(CORPUS, max_words=60)
    assert set(found) == {
        "refunds.md",
        "shipping.html",
        "escalation.pdf",
    }
    assert all(chunks for chunks in found.values())
