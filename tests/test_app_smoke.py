"""Smoke test for the Gradio UI (app.py).

Skipped automatically in CI because ``gradio`` is an optional / soft
dependency and is not installed by ``requirements.txt``.  Locally,
where ``gradio`` is present, this checks that ``app.py`` imports
cleanly and exposes a Gradio interface.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_app_module():
    pytest.importorskip("gradio")
    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "app.py"
    if not app_path.exists():
        pytest.skip("app.py not found")
    spec = importlib.util.spec_from_file_location("aiv_dse_app", app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aiv_dse_app"] = module
    spec.loader.exec_module(module)
    return module


def test_app_imports_cleanly():
    """app.py should import without raising."""
    module = _load_app_module()
    assert module is not None


def test_app_exposes_gradio_interface():
    """app.py should construct a Gradio Blocks / Interface object."""
    gr = pytest.importorskip("gradio")
    module = _load_app_module()
    interfaces = [
        v for v in vars(module).values()
        if isinstance(v, (gr.Blocks, gr.Interface))
    ]
    assert interfaces, "expected app.py to define a Gradio Blocks or Interface"
