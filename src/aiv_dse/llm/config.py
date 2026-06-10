"""LLM configuration and factory for AIV-DSE.

Supports two SDK modes:
  - "langchain"  : LangChain wrappers (ChatOpenAI / ChatAnthropic)
  - "anthropic"  : Direct Anthropic Python SDK (no LangChain)

Set AIVDSE_SDK_MODE=anthropic to use the Anthropic SDK directly.
Default is "langchain" for OpenAI compatibility.

Why both?
  LangChain gives you model-agnostic code -- swap providers with an env var.
  The Anthropic SDK gives you direct control, fewer dependencies, and
  teaches you how the raw API works (messages, tool_use, structured output).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMSettings:
    provider: str = "openai"        # "openai" | "anthropic" | "google"
    model_name: str = "gpt-4o-mini"
    sdk_mode: str = "langchain"     # "langchain" | "anthropic"
    max_retries: int = 2
    log_llm_io: bool = False
    log_dir: str = "./out/debug_llm"

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            provider=os.getenv("AIVDSE_LLM_PROVIDER", "openai"),
            model_name=os.getenv("AIVDSE_MODEL_NAME", "gpt-4o-mini"),
            sdk_mode=os.getenv("AIVDSE_SDK_MODE", "langchain"),
            max_retries=int(os.getenv("AIVDSE_LLM_MAX_RETRIES", "2")),
            log_llm_io=os.getenv("AIVDSE_LOG_LLM_IO", "0") == "1",
            log_dir=os.getenv("AIVDSE_LLM_LOG_DIR", "./out/debug_llm"),
        )


# ---------------------------------------------------------------------------
# Judge provider selection (Phase B3: adversarial judge across providers)
# ---------------------------------------------------------------------------
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash-exp",
}

_PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def get_judge_settings(advisor: LLMSettings) -> LLMSettings:
    """Return LLM settings for the judge, defaulting to a DIFFERENT provider
    than the advisor for adversarial diversity.

    Precedence:
      1. AIVDSE_JUDGE_PROVIDER env var overrides everything.
      2. Otherwise, pick the "opposite" provider if its API key is set:
         anthropic advisor -> google (preferred) -> openai -> same
         openai advisor    -> anthropic -> google -> same
         google advisor    -> anthropic -> openai -> same
      3. Fall back to same provider as advisor (preserves legacy behaviour
         when only one API key is configured).

    The judge always uses LangChain SDK mode (uniformly supports all providers).
    """
    override = os.getenv("AIVDSE_JUDGE_PROVIDER")
    if override:
        return _build_judge_settings(override, advisor)

    # Auto-pick opposite provider if its key is available
    preferred_order = {
        "anthropic": ["google", "openai"],
        "openai": ["anthropic", "google"],
        "google": ["anthropic", "openai"],
    }.get(advisor.provider, [])

    for candidate in preferred_order:
        key_name = _PROVIDER_API_KEYS.get(candidate)
        if key_name and os.getenv(key_name):
            return _build_judge_settings(candidate, advisor)

    # No alternative configured - fall back to same provider
    return advisor


def _build_judge_settings(provider: str, advisor: LLMSettings) -> LLMSettings:
    """Construct judge settings inheriting log/retry config from the advisor."""
    model = os.getenv("AIVDSE_JUDGE_MODEL_NAME") or _DEFAULT_MODELS.get(
        provider, "gpt-4o-mini"
    )
    # Cross-provider judge always uses LangChain SDK uniformly
    sdk = "langchain" if provider != advisor.provider else advisor.sdk_mode
    return LLMSettings(
        provider=provider,
        model_name=model,
        sdk_mode=sdk,
        max_retries=advisor.max_retries,
        log_llm_io=advisor.log_llm_io,
        log_dir=advisor.log_dir,
    )


# ---------------------------------------------------------------------------
# LangChain factory
# ---------------------------------------------------------------------------
def get_llm_langchain(settings: LLMSettings):
    """Return a LangChain BaseChatModel.

    LangChain abstracts the provider behind a common interface.
    Switching from OpenAI to Anthropic is just an env-var change --
    your calling code stays identical.
    """
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.model_name, temperature=0)

    elif settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.model_name, temperature=0)

    elif settings.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.model_name, temperature=0)

    raise ValueError(f"Unsupported provider for langchain mode: {settings.provider}")


# ---------------------------------------------------------------------------
# Direct Anthropic SDK factory
# ---------------------------------------------------------------------------
def get_anthropic_client(settings: LLMSettings):
    """Return a raw anthropic.Anthropic client.

    This is the direct SDK -- no LangChain wrapper.  You construct
    messages yourself, call client.messages.create(), and parse the
    response manually.  More verbose, but you see exactly what goes
    over the wire.

    Requires: pip install anthropic
    Requires: ANTHROPIC_API_KEY env var
    """
    import anthropic
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


# ---------------------------------------------------------------------------
# Unified factory (dispatches on sdk_mode)
# ---------------------------------------------------------------------------
def get_llm(settings: LLMSettings):
    """Return an LLM client based on sdk_mode.

    sdk_mode="langchain" -> LangChain BaseChatModel
    sdk_mode="anthropic" -> anthropic.Anthropic client
    """
    if settings.sdk_mode == "langchain":
        return get_llm_langchain(settings)
    elif settings.sdk_mode == "anthropic":
        return get_anthropic_client(settings)
    else:
        raise ValueError(f"Unsupported sdk_mode: {settings.sdk_mode}")
