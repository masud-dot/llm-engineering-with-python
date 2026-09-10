"""The techniques of Chapter 18, tested on themselves."""

import json
import shutil
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llmapp.llm.base import Message
from llmapp.llm.fake import ScriptedClient
from llmapp.llm.openai_adapter import OpenAIClient
from llmapp.retrieval.chunking import chunk_text, split_sentences
from tests import _cassette
from tests._cassette import key_for, serve
from tests._mock_openai import serve as serve_upstream
from tests._statistics import (
    assert_pass_rate,
    repeat,
    within,
)

ASK = [Message(role="user", content="classify this ticket")]


# --- 18.4 test doubles ----------------------------------------


def test_a_scripted_client_satisfies_the_port() -> None:
    client = ScriptedClient(["billing"])
    assert client.complete(ASK).text == "billing"


def test_a_double_records_what_it_was_sent() -> None:
    client = ScriptedClient(["billing"])
    client.complete(ASK)
    assert client.calls[0][0].content == "classify this ticket"


def test_running_out_of_script_is_an_error() -> None:
    client = ScriptedClient(["one"])
    client.complete(ASK)
    with pytest.raises(AssertionError, match="more calls"):
        client.complete(ASK)


# --- 18.5 record and replay -----------------------------------


@pytest.fixture
def cassettes(tmp_path: Path) -> Path:
    original = _cassette.CASSETTES
    _cassette.CASSETTES = tmp_path / "cassettes"
    yield _cassette.CASSETTES
    _cassette.CASSETTES = original


def test_the_key_ignores_irrelevant_fields() -> None:
    left = {"model": "m", "input": "hello", "temperature": 0.1}
    right = {"model": "m", "input": "hello", "temperature": 0.9}
    assert key_for("/v1/responses", left) == key_for(
        "/v1/responses", right
    )


def test_the_key_separates_different_requests() -> None:
    left = {"model": "m", "input": "hello"}
    right = {"model": "m", "input": "goodbye"}
    assert key_for("/v1/responses", left) != key_for(
        "/v1/responses", right
    )


@pytest.mark.local_server
def test_record_then_replay_without_the_upstream(
    cassettes: Path,
) -> None:
    serve_upstream(8091)
    serve(8090, mode="record", upstream="http://127.0.0.1:8091")
    recorder = OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8090/v1",
    )
    first = recorder.complete(ASK)
    assert first.text == "billing"
    written = list(cassettes.glob("*.json"))
    assert len(written) == 1

    # Replay talks to nothing but the files on disk.
    serve(8089, mode="replay")
    player = OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8089/v1",
    )
    assert player.complete(ASK).text == "billing"


@pytest.mark.local_server
def test_an_unrecorded_request_fails_loudly(
    cassettes: Path,
) -> None:
    cassettes.mkdir(parents=True, exist_ok=True)
    serve(8088, mode="replay")
    player = OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8088/v1",
    )
    with pytest.raises(Exception, match="no cassette"):
        player.complete(
            [Message(role="user", content="never recorded")]
        )


# --- 18.8 property tests --------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    sentences=st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu")
            ),
            min_size=3,
            max_size=12,
        ),
        min_size=1,
        max_size=20,
    )
)
def test_chunking_never_loses_a_sentence(
    sentences: list[str]
) -> None:
    text = ". ".join(sentences) + "."
    chunks = chunk_text(
        text, "doc.md", max_words=5, overlap_words=2
    )
    joined = " ".join(c.text for c in chunks)
    for sentence in split_sentences(text):
        assert sentence in joined


@settings(max_examples=30, deadline=None)
@given(
    sentences=st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll",)
            ),
            min_size=3,
            max_size=10,
        ),
        min_size=1,
        max_size=15,
    )
)
def test_every_chunk_has_provenance(
    sentences: list[str]
) -> None:
    text = ". ".join(sentences) + "."
    chunks = chunk_text(text, "doc.md", max_words=6, overlap_words=2)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "doc.md" for c in chunks)


# --- 18.9 and 18.10 assertions --------------------------------


def test_repeat_counts_outcomes() -> None:
    results = iter([True, False, True, True])
    trials = repeat(lambda: next(results), times=4)
    assert trials.passed == 3
    assert trials.rate == 0.75


def test_a_pass_rate_above_the_floor_succeeds() -> None:
    results = iter([True] * 19 + [False])
    trials = assert_pass_rate(
        lambda: next(results), times=20, minimum=0.9
    )
    assert trials.rate == 0.95


def test_a_pass_rate_below_the_floor_fails() -> None:
    results = iter([True] * 15 + [False] * 5)
    with pytest.raises(AssertionError, match="15/20"):
        assert_pass_rate(
            lambda: next(results), times=20, minimum=0.9
        )


def test_a_tolerance_band_accepts_close_values() -> None:
    assert within(0.83, 0.85, band=0.05)
    assert not within(0.70, 0.85, band=0.05)


def test_zero_trials_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        repeat(lambda: True, times=0)
