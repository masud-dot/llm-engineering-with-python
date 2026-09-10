import pytest
from pydantic import ValidationError

from llmapp.config import Settings


def test_defaults_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMAPP_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.provider == "fake"
    assert settings.environment == "local"


def test_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMAPP_REQUEST_TIMEOUT_S", "5.5")
    settings = Settings(_env_file=None)
    assert settings.request_timeout_s == 5.5


def test_real_provider_requires_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMAPP_PROVIDER", "openai")
    monkeypatch.delenv("LLMAPP_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="LLMAPP_API_KEY"):
        Settings(_env_file=None)


def test_negative_timeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMAPP_REQUEST_TIMEOUT_S", "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_key_is_not_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMAPP_PROVIDER", "openai")
    monkeypatch.setenv("LLMAPP_MODEL", "some-model")
    monkeypatch.setenv("LLMAPP_API_KEY", "sk-do-not-log-me")
    settings = Settings(_env_file=None)
    assert "do-not-log-me" not in repr(settings)
    assert settings.api_key is not None
    assert (
        settings.api_key.get_secret_value() == "sk-do-not-log-me"
    )
