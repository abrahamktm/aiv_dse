"""Tests for SystemC code analyzer."""

import pytest

from aiv_dse.core.code_analyzer import analyze_source
from aiv_dse.llm.models import CodeProfile


def test_analyze_fft_sample():
    """Analyze the sample FFT file and verify profile structure."""
    profile = analyze_source("samples/fft256_design.cpp")
    assert isinstance(profile, CodeProfile)
    assert profile.total_lines > 50
    assert len(profile.loops) >= 4      # outer + inner butterfly + bit-reverse + output
    assert len(profile.arrays) >= 4     # buf_re, buf_im, twiddle_re, twiddle_im
    assert len(profile.pragmas) >= 1    # at least the existing PIPELINE
    assert len(profile.functions) >= 2  # fft256 + butterfly_op


def test_loop_for_extraction():
    """For-loop iteration count is parsed correctly."""
    profile = analyze_source("samples/fft256_design.cpp")
    # Find the 256-iteration loops
    big_loops = [l for l in profile.loops if l.iteration_count == 256]
    assert len(big_loops) >= 1


def test_loop_pipeline_pragma_detected():
    """Pipeline pragma on inner loop is detected."""
    profile = analyze_source("samples/fft256_design.cpp")
    pipelined = [l for l in profile.loops if l.has_pipeline_pragma]
    assert len(pipelined) >= 1


def test_array_dimensions():
    """Array dimensions are extracted correctly."""
    profile = analyze_source("samples/fft256_design.cpp")
    # Find the 256-element arrays
    big_arrays = [a for a in profile.arrays if 256 in a.dimensions]
    assert len(big_arrays) >= 2  # buf_re, buf_im (and possibly in/out)

    # Find the 128-element twiddle arrays
    twiddle = [a for a in profile.arrays if 128 in a.dimensions]
    assert len(twiddle) >= 2  # twiddle_re, twiddle_im


def test_pragma_categorization():
    """Pragmas are categorized correctly."""
    profile = analyze_source("samples/fft256_design.cpp")
    categories = {p.category for p in profile.pragmas}
    assert "pipeline" in categories


def test_function_extraction():
    """Functions are extracted with names."""
    profile = analyze_source("samples/fft256_design.cpp")
    names = {f.name for f in profile.functions}
    assert "butterfly_op" in names
    assert "fft256" in names


def test_memory_pattern():
    """Memory pattern is inferred as something meaningful."""
    profile = analyze_source("samples/fft256_design.cpp")
    assert profile.memory_access_pattern in ("sequential", "strided", "random", "unknown")


def test_empty_file(tmp_path):
    """Empty file produces empty CodeProfile."""
    empty = tmp_path / "empty.cpp"
    empty.write_text("")
    profile = analyze_source(str(empty))
    assert profile.total_lines == 1  # One empty line
    assert len(profile.loops) == 0
    assert len(profile.arrays) == 0


def test_file_not_found():
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        analyze_source("nonexistent_file.cpp")
