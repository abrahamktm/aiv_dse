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
    provider: str = "openai"        # "openai" | "anthropic"
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
