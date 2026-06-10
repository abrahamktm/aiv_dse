"""Tests for the tracing module."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestTracingModule:
    """Test the tracing module's observe decorator and helpers."""

    def test_observe_noop_when_disabled(self):
        """When AIVDSE_USE_LANGFUSE=0, @observe is a no-op."""
        # Ensure Langfuse is disabled
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            # Re-import to pick up env change
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            @tracing_module.observe(name="test_func")
            def my_func(x):
                return x * 2

            # Should work normally
            assert my_func(5) == 10
            assert my_func.__name__ == "my_func"

    def test_observe_preserves_return_value(self):
        """Decorated function returns the correct value."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            @tracing_module.observe(name="adder")
            def add(a, b):
                return a + b

            assert add(3, 4) == 7

    def test_observe_preserves_exceptions(self):
        """Decorated function preserves exceptions."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            @tracing_module.observe(name="raiser")
            def raise_error():
                raise ValueError("test error")

            with pytest.raises(ValueError, match="test error"):
                raise_error()

    def test_get_langfuse_returns_none_when_disabled(self):
        """get_langfuse() returns None when disabled."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            assert tracing_module.get_langfuse() is None

    def test_flush_traces_noop_when_disabled(self):
        """flush_traces() is a no-op when Langfuse is disabled."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            # Should not raise
            tracing_module.flush_traces()

    def test_trace_llm_call_noop_when_disabled(self):
        """trace_llm_call() is a no-op when Langfuse is disabled."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            # Should not raise
            tracing_module.trace_llm_call(
                name="test",
                model="gpt-4o-mini",
                input_text="hello",
                output_text="world",
            )

    def test_create_trace_returns_none_when_disabled(self):
        """create_trace() returns None when Langfuse is disabled."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            trace = tracing_module.create_trace("test_trace")
            assert trace is None

    def test_observe_with_langfuse_enabled_but_not_installed(self, capfd):
        """When enabled but langfuse not installed, prints warning and works."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "1"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            # Reset the singleton
            tracing_module._langfuse_client = None
            importlib.reload(tracing_module)

            # Mock ImportError for langfuse
            with patch.dict("sys.modules", {"langfuse": None, "langfuse.decorators": None}):
                @tracing_module.observe(name="test_func")
                def my_func(x):
                    return x + 1

                # Should still work
                assert my_func(5) == 6


class TestTracingIntegrationWithAdvisors:
    """Test that @observe decorator works with actual advisor functions."""

    def test_observe_on_constraint_advisor_import(self):
        """Importing constraint_advisor with tracing disabled works."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            # Should import without error
            from aiv_dse.llm import constraint_advisor
            importlib.reload(constraint_advisor)

            # The function should exist and be callable (though it needs real args)
            assert hasattr(constraint_advisor, "propose_adjustments")

    def test_observe_on_synth_advisor_import(self):
        """Importing synth_advisor with tracing disabled works."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            from aiv_dse.llm import synth_advisor
            importlib.reload(synth_advisor)

            assert hasattr(synth_advisor, "propose_synth_params")

    def test_observe_on_judge_import(self):
        """Importing judge with tracing disabled works."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            from aiv_dse.llm import judge
            importlib.reload(judge)

            assert hasattr(judge, "judge_proposal")
            assert hasattr(judge, "judge_code_advisory")

    def test_observe_on_spec_planner_import(self):
        """Importing spec_planner with tracing disabled works."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            from aiv_dse.llm import spec_planner
            importlib.reload(spec_planner)

            assert hasattr(spec_planner, "plan_from_spec")

    def test_observe_on_code_advisor_import(self):
        """Importing code_advisor with tracing disabled works."""
        with patch.dict(os.environ, {"AIVDSE_USE_LANGFUSE": "0"}):
            import importlib
            import aiv_dse.tracing as tracing_module
            importlib.reload(tracing_module)

            from aiv_dse.llm import code_advisor
            importlib.reload(code_advisor)

            assert hasattr(code_advisor, "advise_code_changes")
