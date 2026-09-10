import pytest

from llmapp.retrieval.chunking import chunk_text, split_sentences

TEXT = (
    "Refunds are issued within fourteen days. "
    "Damaged items qualify for a replacement. "
    "Express delivery is refunded separately. "
    "Contact support with the order number. "
    "Claims after ninety days are declined."
)


def test_sentences_are_split_on_terminators() -> None:
    assert len(split_sentences(TEXT)) == 5


def test_small_documents_make_one_chunk() -> None:
    chunks = chunk_text(TEXT, "policy.md", max_words=200)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "policy.md#0"


def test_chunks_respect_the_word_budget() -> None:
    chunks = chunk_text(
        TEXT, "policy.md", max_words=12, overlap_words=4
    )
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(chunk.text.split()) <= 12 + 6


def test_overlap_repeats_the_trailing_sentence() -> None:
    chunks = chunk_text(
        TEXT, "policy.md", max_words=12, overlap_words=8
    )
    first_end = chunks[0].text.split(". ")[-1]
    assert first_end.rstrip(".") in chunks[1].text


def test_no_sentence_is_lost() -> None:
    chunks = chunk_text(
        TEXT, "policy.md", max_words=12, overlap_words=4
    )
    joined = " ".join(c.text for c in chunks)
    for sentence in split_sentences(TEXT):
        assert sentence in joined


def test_metadata_is_carried_to_every_chunk() -> None:
    chunks = chunk_text(
        TEXT,
        "policy.md",
        max_words=12,
        overlap_words=4,
        metadata={"team": "support", "visibility": "internal"},
    )
    assert all(c.metadata["team"] == "support" for c in chunks)


def test_overlap_must_be_smaller_than_the_chunk() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(TEXT, "x", max_words=10, overlap_words=10)
