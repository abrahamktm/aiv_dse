"""Tests for B3: adversarial judge across providers.

Verifies that get_judge_settings() returns a *different* provider than the
advisor when an alternative provider's API key is configured -- and falls
back to the same provider when no alternative is available.
"""

import os

import pytest

from aiv_dse.llm.config import LLMSettings, get_judge_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Wipe relevant env vars so each test starts from a known state."""
    for var in (
        "AIVDSE_JUDGE_PROVIDER",
        "AIVDSE_JUDGE_MODEL_NAME",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _advisor(provider="anthropic"):
    return LLMSettings(
        provider=provider,
        model_name="test-model",
        sdk_mode="anthropic",
        max_retries=1,
    )


class TestExplicitOverride:
    def test_judge_provider_env_overrides(self, monkeypatch):
        monkeypatch.setenv("AIVDSE_JUDGE_PROVIDER", "openai")
        advisor = _advisor("anthropic")
        judge = get_judge_settings(advisor)
        assert judge.provider == "openai"

    def test_judge_model_env_used_when_override_set(self, monkeypatch):
        monkeypatch.setenv("AIVDSE_JUDGE_PROVIDER", "google")
        monkeypatch.setenv("AIVDSE_JUDGE_MODEL_NAME", "gemini-custom")
        advisor = _advisor("anthropic")
        judge = get_judge_settings(advisor)
        assert judge.provider == "google"
        assert judge.model_name == "gemini-custom"

    def test_judge_uses_default_model_when_only_provider_overridden(self, monkeypatch):
        monkeypatch.setenv("AIVDSE_JUDGE_PROVIDER", "google")
        advisor = _advisor("anthropic")
        judge = get_judge_settings(advisor)
        assert judge.model_name == "gemini-2.0-flash-exp"


class TestAutoOppositeProvider:
    """The headline behaviour: Claude advisor -> Gemini judge by default."""

    def test_anthropic_advisor_prefers_google_judge(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        advisor = _advisor("anthropic")
        judge = get_judge_settings(advisor)
        assert judge.provider == "google"
        # Cross-provider judge always switches to langchain for uniform support
        assert judge.sdk_mode == "langchain"

    def test_anthropic_advisor_falls_back_to_openai_when_no_google(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        advisor = _advisor("anthropic")
        judge = get_judge_settings(advisor)
        assert judge.provider == "openai"

    def test_openai_advisor_prefers_anthropic_judge(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        advisor = _advisor("openai")
        judge = get_judge_settings(advisor)
        assert judge.provider == "anthropic"

    def test_falls_back_to_same_provider_when_no_alternative(self, monkeypatch):
        # No alternative API keys set
        advisor = _advisor("anthropic")
        judge = get_judge_settings(advisor)
        # Same provider as advisor
        assert judge.provider == "anthropic"
        # And inherits the advisor's sdk_mode (no cross-provider needed)
        assert judge.sdk_mode == "anthropic"

    def test_judge_inherits_advisor_log_settings(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        advisor = LLMSettings(
            provider="anthropic",
            sdk_mode="anthropic",
            log_llm_io=True,
            log_dir="/tmp/custom",
        )
        judge = get_judge_settings(advisor)
        assert judge.log_llm_io is True
        assert judge.log_dir == "/tmp/custom"
