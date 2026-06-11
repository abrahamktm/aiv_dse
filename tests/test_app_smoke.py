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


@pytest.fixture(scope="module")
def app_module():
    pytest.importorskip("gradio")
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    if not app_path.exists():
        pytest.skip("app.py not found")

    spec = importlib.util.spec_from_file_location("aiv_dse_app", app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aiv_dse_app"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("aiv_dse_app", None)


def test_app_exposes_gradio_interface(app_module):
    """app.py should construct a Gradio Blocks / Interface object."""
    gr = pytest.importorskip("gradio")
    interfaces = [
        v for v in vars(app_module).values()
        if isinstance(v, (gr.Blocks, gr.Interface))
    ]
    assert interfaces, "expected app.py to define a Gradio Blocks or Interface"
