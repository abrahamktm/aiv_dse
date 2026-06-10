"""Tracing module for AIV-DSE — routes to Langfuse when enabled.

Langfuse provides LLM observability: traces, spans, latency metrics, and
a dashboard to visualize the adversarial validation flow (primary LLM →
judge LLM → escalation decision).

Usage:
    from aiv_dse.tracing import observe, get_langfuse, flush_traces

    @observe(name="synth_advisor")
    def propose_synth_params(...):
        ...

Environment variables:
    AIVDSE_USE_LANGFUSE=1           # Enable Langfuse tracing (default: 0)
    LANGFUSE_SECRET_KEY=sk-lf-...   # Langfuse secret key
    LANGFUSE_PUBLIC_KEY=pk-lf-...   # Langfuse public key
    LANGFUSE_HOST=https://cloud.langfuse.com  # Or self-hosted URL

When AIVDSE_USE_LANGFUSE=0 (default), the @observe decorator is a no-op.
Local file logging (AIVDSE_LOG_LLM_IO) continues to work independently.
"""

import os
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from dotenv import load_dotenv

load_dotenv()

# Check if Langfuse is enabled
USE_LANGFUSE = os.getenv("AIVDSE_USE_LANGFUSE", "0") == "1"

# Type variable for preserving function signatures
F = TypeVar("F", bound=Callable[..., Any])

# Langfuse client singleton (lazy init)
_langfuse_client: Optional[Any] = None


def get_langfuse():
    """Get the Langfuse client singleton.

    Returns None if Langfuse is not enabled or not installed.
    """
    global _langfuse_client

    if not USE_LANGFUSE:
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse()
        return _langfuse_client
    except ImportError:
        print(
            "Warning: AIVDSE_USE_LANGFUSE=1 but langfuse not installed. "
            "Install with: pip install langfuse"
        )
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse: {e}")
        return None


def flush_traces() -> None:
    """Flush any pending Langfuse traces.

    Call this at the end of a run to ensure all traces are sent.
    """
    if not USE_LANGFUSE:
        return

    try:
        from langfuse import get_client
        client = get_client()
        client.flush()
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to flush Langfuse traces: {e}")


def observe(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to trace function execution with Langfuse.

    When AIVDSE_USE_LANGFUSE=1, wraps the function with Langfuse's
    @observe decorator for automatic span creation and timing.

    When AIVDSE_USE_LANGFUSE=0 (default), this is a no-op decorator
    that returns the original function unchanged.

    Args:
        name: Optional span name. Defaults to the function name.

    Returns:
        Decorated function (traced if Langfuse enabled, unchanged otherwise).

    Example:
        @observe(name="constraint_advisor")
        def propose_adjustments(policy, state, result, settings):
            ...
    """
    if USE_LANGFUSE:
        try:
            # Langfuse v4+ has observe in main module
            from langfuse import observe as langfuse_observe

            # Langfuse's observe decorator
            return langfuse_observe(name=name)
        except ImportError:
            # Langfuse not installed, fall through to no-op
            pass

    # No-op decorator when Langfuse is disabled or not installed
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def trace_llm_call(
    name: str,
    model: str,
    input_text: str,
    output_text: str,
    metadata: Optional[dict] = None,
) -> None:
    """Manually trace an LLM call to Langfuse.

    Use this for cases where the @observe decorator isn't suitable,
    such as when you want to trace just the LLM API call portion
    of a larger function.

    Args:
        name: Trace name (e.g., "synth_advisor", "judge_crosscheck").
        model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514").
        input_text: The prompt/input sent to the LLM.
        output_text: The response/output from the LLM.
        metadata: Optional additional metadata dict.
    """
    client = get_langfuse()
    if client is None:
        return

    try:
        trace = client.trace(name=name, metadata=metadata or {})
        trace.generation(
            name=f"{name}_generation",
            model=model,
            input=input_text,
            output=output_text,
            metadata=metadata or {},
        )
    except Exception as e:
        print(f"Warning: Failed to trace LLM call to Langfuse: {e}")


def create_trace(name: str, metadata: Optional[dict] = None):
    """Create a new Langfuse trace for a multi-step operation.

    Use this to group related LLM calls (e.g., primary + judge) under
    a single trace.

    Args:
        name: Trace name (e.g., "adversarial_validation").
        metadata: Optional metadata dict.

    Returns:
        Langfuse trace object, or None if Langfuse is disabled.

    Example:
        trace = create_trace("adversarial_validation")
        if trace:
            span1 = trace.span(name="primary_proposal")
            # ... do primary LLM call ...
            span1.end()

            span2 = trace.span(name="judge_crosscheck")
            # ... do judge LLM call ...
            span2.end()
    """
    client = get_langfuse()
    if client is None:
        return None

    try:
        return client.trace(name=name, metadata=metadata or {})
    except Exception as e:
        print(f"Warning: Failed to create Langfuse trace: {e}")
        return None
