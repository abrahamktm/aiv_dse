"""Tests for B2: extended thinking on judge.

Verifies the env-var-gated extended-thinking branch in the Anthropic SDK path.
We don't make real API calls -- we patch anthropic.Anthropic to capture the
kwargs the judge sends and assert they include the thinking parameter.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from aiv_dse.core.validator import ValidationResult
from aiv_dse.llm.config import LLMSettings
from aiv_dse.llm.judge import _judge_via_anthropic
from aiv_dse.llm.models import (
    SynthParamAdjustment,
    SynthParamProposal,
    SynthesisParams,
)


@pytest.fixture
def fake_anthropic_response():
    """Build a fake Anthropic SDK response with a tool_use block matching JudgeVerdict."""
    response = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "agree": True,
        "disagreements": [],
        "alternative_suggestion": "",
        "confidence": 0.9,
    }
    response.content = [tool_block]
    return response


def _build_settings():
    return LLMSettings(
        provider="anthropic",
        model_name="claude-sonnet-4-5-20250929",
        sdk_mode="anthropic",
        max_retries=1,
    )


class TestPromptCachingApplied:
    """B1 verification: system prompt is structured with cache_control."""

    def test_system_prompt_has_cache_control(self, monkeypatch, fake_anthropic_response):
        monkeypatch.delenv("AIVDSE_JUDGE_THINKING", raising=False)
        with patch("anthropic.Anthropic") as fake_client_cls:
            client = fake_client_cls.return_value
            client.messages.create.return_value = fake_anthropic_response
            _judge_via_anthropic("ctx", _build_settings())
            kwargs = client.messages.create.call_args.kwargs
            # Phase B1: system should be a list with cache_control set
            assert isinstance(kwargs["system"], list)
            assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


class TestExtendedThinkingDisabledByDefault:
    def test_no_thinking_param_when_env_not_set(self, monkeypatch, fake_anthropic_response):
        monkeypatch.delenv("AIVDSE_JUDGE_THINKING", raising=False)
        with patch("anthropic.Anthropic") as fake_client_cls:
            client = fake_client_cls.return_value
            client.messages.create.return_value = fake_anthropic_response
            _judge_via_anthropic("ctx", _build_settings())
            kwargs = client.messages.create.call_args.kwargs
            assert "thinking" not in kwargs
            # tool_choice should still be forced (specific tool) in default mode
            assert kwargs["tool_choice"] == {"type": "tool", "name": "judge_verdict"}


class TestExtendedThinkingEnabled:
    def test_thinking_param_added_when_env_set(self, monkeypatch, fake_anthropic_response):
        monkeypatch.setenv("AIVDSE_JUDGE_THINKING", "1")
        with patch("anthropic.Anthropic") as fake_client_cls:
            client = fake_client_cls.return_value
            client.messages.create.return_value = fake_anthropic_response
            _judge_via_anthropic("ctx", _build_settings())
            kwargs = client.messages.create.call_args.kwargs
            assert "thinking" in kwargs
            assert kwargs["thinking"]["type"] == "enabled"
            assert kwargs["thinking"]["budget_tokens"] >= 1024
            # Forced tool_choice incompatible with thinking; should be auto
            assert kwargs["tool_choice"] == {"type": "auto"}
            # Temperature must be 1.0 when thinking is enabled
            assert kwargs["temperature"] == 1.0

    def test_thinking_budget_configurable_via_env(self, monkeypatch, fake_anthropic_response):
        monkeypatch.setenv("AIVDSE_JUDGE_THINKING", "1")
        monkeypatch.setenv("AIVDSE_JUDGE_THINKING_BUDGET", "8192")
        with patch("anthropic.Anthropic") as fake_client_cls:
            client = fake_client_cls.return_value
            client.messages.create.return_value = fake_anthropic_response
            _judge_via_anthropic("ctx", _build_settings())
            kwargs = client.messages.create.call_args.kwargs
            assert kwargs["thinking"]["budget_tokens"] == 8192
