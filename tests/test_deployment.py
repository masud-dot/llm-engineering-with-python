"""The code the container runs, tested without a container."""

import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from llmapp.api.run import build_embedder, build_services
from llmapp.config import Settings
from llmapp.retrieval.local import HashEmbedder

ROOT = Path(".")


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "provider": "fake",
        "jwt_secret": "a-secret-that-is-at-least-32-bytes-long",
        "corpus_dir": Path("tests/corpus"),
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# --- the local embedder ---------------------------------------


def test_the_local_embedder_is_deterministic() -> None:
    one = HashEmbedder(64).embed(["refund policy"])
    two = HashEmbedder(64).embed(["refund policy"])
    assert np.array_equal(one, two)


def test_it_never_returns_a_zero_vector() -> None:
    vectors = HashEmbedder(64).embed(["", "   "])
    assert np.all(np.linalg.norm(vectors, axis=1) > 0)


def test_it_respects_the_requested_width() -> None:
    assert HashEmbedder(128).embed(["a", "b"]).shape == (2, 128)


def test_a_useless_width_is_refused() -> None:
    with pytest.raises(ValueError, match="too few dimensions"):
        HashEmbedder(4)


def test_identical_text_scores_higher_than_unrelated() -> None:
    embedder = HashEmbedder(256)
    rows = embedder.embed(
        ["refund policy fourteen days", "refund policy", "x y z"]
    )
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    assert unit[0] @ unit[1] > unit[0] @ unit[2]


# --- the startup path -----------------------------------------


def test_the_fake_provider_embeds_locally() -> None:
    """The defect this test exists for: the SDK refuses to
    construct without credentials, so the provider check has
    to be in the embedder factory too."""
    assert isinstance(build_embedder(settings()), HashEmbedder)


def test_services_are_constructed_without_credentials() -> None:
    services = build_services(settings())
    assert services.pipeline is not None
    assert len(services.pipeline._index) > 0


def test_an_absent_corpus_is_not_fatal() -> None:
    services = build_services(
        settings(corpus_dir=Path("nowhere"))
    )
    assert len(services.pipeline._index) == 0


# --- production configuration ---------------------------------


def test_production_requires_a_jwt_secret() -> None:
    with pytest.raises(Exception, match="LLMAPP_JWT_SECRET"):
        Settings(
            _env_file=None,
            environment="production",
            provider="openai",
            model="m",
            api_key="k",
        )


def test_production_refuses_a_short_secret() -> None:
    with pytest.raises(Exception, match="32 bytes"):
        Settings(
            _env_file=None,
            environment="production",
            provider="openai",
            model="m",
            api_key="k",
            jwt_secret="too-short",
        )


def test_production_refuses_the_fake_provider() -> None:
    with pytest.raises(Exception, match="fake provider"):
        Settings(
            _env_file=None,
            environment="production",
            provider="fake",
            jwt_secret="a-secret-that-is-at-least-32-bytes-long",
        )


# --- the build artifacts --------------------------------------


def test_the_declared_pins_match_the_lockfile() -> None:
    """The drift this test exists for was real."""
    declared = tomllib.loads(
        (ROOT / "pyproject.toml").read_text()
    )["project"]["dependencies"]
    pinned = {
        line.split("==")[0].lower(): line.split("==")[1]
        for line in (ROOT / "requirements.txt")
        .read_text()
        .splitlines()
        if "==" in line
    }
    for entry in declared:
        if "==" not in entry:
            continue
        name, version = entry.split("==")
        assert pinned.get(name.lower()) == version, name


def test_the_dockerfile_runs_as_a_non_root_user() -> None:
    text = (ROOT / "Dockerfile").read_text()
    assert "USER llmapp" in text
    assert text.index("USER llmapp") < text.index("CMD")


def test_the_dockerignore_excludes_secrets_and_tests() -> None:
    text = (ROOT / ".dockerignore").read_text().splitlines()
    for entry in (".env", "tests", ".git"):
        assert entry in text


def test_the_readiness_audit_fails_when_misconfigured() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/readiness_audit.py"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
    )
    assert result.returncode == 1
    assert "LLMAPP_JWT_SECRET" in result.stdout
