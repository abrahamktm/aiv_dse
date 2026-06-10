"""Regex-based static analysis of SystemC/C++ source files.

Extracts loops, arrays, pragmas, functions, and memory access patterns.
Best-effort: missing extractions produce fewer suggestions, not errors.
"""

import re
from typing import List, Optional, Tuple

from aiv_dse.llm.models import (
    ArrayInfo,
    CodeProfile,
    FunctionInfo,
    LoopInfo,
    PragmaInfo,
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# for (int i = 0; i < N; i++)  or  for (int i = 0; i <= N; i++)
RE_FOR_LOOP = re.compile(
    r'^\s*for\s*\(\s*'
    r'(?:int|unsigned|size_t|auto)\s+(\w+)\s*=\s*(\d+)\s*;\s*'
    r'\w+\s*(<|<=)\s*(\w+)\s*;',
    re.MULTILINE,
)

# Simple while loop detection
RE_WHILE_LOOP = re.compile(r'^\s*while\s*\(', re.MULTILINE)

# Array declarations: type name[N] or type name[N][M]
# Handles sc_fixed<W,I>, ac_int<W>, int, float, etc.
RE_ARRAY_DECL = re.compile(
    r'^\s*'
    r'((?:static\s+)?(?:const\s+)?'  # optional static/const
    r'(?:sc_fixed|sc_ufixed|ac_int|ac_fixed|int|unsigned|float|double|bool|char|short|long)'
    r'(?:<[^>]*>)?'  # optional template params
    r')\s+'
    r'(\w+)'  # variable name
    r'((?:\[\d+\])+)'  # one or more [N] dimensions
    r'\s*;',
    re.MULTILINE,
)

# HLS pragma
RE_PRAGMA_HLS = re.compile(
    r'^\s*#pragma\s+HLS\s+(\w+)(.*)',
    re.MULTILINE,
)

# Function definitions
RE_FUNC_DEF = re.compile(
    r'^\s*(?:void|int|float|double|bool|auto|'
    r'sc_fixed<[^>]*>|sc_ufixed<[^>]*>|ac_int<[^>]*>|ac_fixed<[^>]*>)'
    r'\s+(\w+)\s*\([^)]*\)\s*\{',
    re.MULTILINE,
)

# Function calls within a body
RE_FUNC_CALL = re.compile(r'\b(\w+)\s*\(')

# Standard library / language keywords to exclude from call graph
_BUILTIN_NAMES = {
    "if", "for", "while", "switch", "return", "sizeof", "static_cast",
    "dynamic_cast", "reinterpret_cast", "const_cast", "printf", "cout",
    "assert", "int", "float", "double", "bool", "void", "auto",
    "true", "false", "NULL", "nullptr",
}


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def _extract_loops(lines: List[str]) -> List[LoopInfo]:
    """Extract for and while loops with nesting depth tracking."""
    loops = []
    text = "\n".join(lines)

    # Track brace depth for nesting
    current_depth = 0
    loop_stack: List[int] = []  # depths where loops start

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Count braces for nesting depth
        current_depth += stripped.count('{') - stripped.count('}')

        # For loops
        m = RE_FOR_LOOP.match(stripped)
        if m:
            var_name = m.group(1)
            start_val = int(m.group(2))
            comparator = m.group(3)
            bound = m.group(4)

            # Try to determine iteration count
            iter_count = None
            try:
                bound_val = int(bound)
                if comparator == "<":
                    iter_count = bound_val - start_val
                elif comparator == "<=":
                    iter_count = bound_val - start_val + 1
            except ValueError:
                pass  # Non-numeric bound (variable)

            # Check for pragmas in nearby lines
            has_pipeline = _check_nearby_pragma(lines, i, "PIPELINE")
            has_unroll = _check_nearby_pragma(lines, i, "UNROLL")

            # Estimate body line count
            body_lines = _count_body_lines(lines, i)

            nesting = len([d for d in loop_stack if d < current_depth])

            loops.append(LoopInfo(
                line_number=i + 1,
                loop_type="for",
                iteration_count=iter_count,
                body_line_count=body_lines,
                has_pipeline_pragma=has_pipeline,
                has_unroll_pragma=has_unroll,
                nesting_depth=nesting,
            ))
            loop_stack.append(current_depth)

        # While loops
        elif RE_WHILE_LOOP.match(stripped):
            has_pipeline = _check_nearby_pragma(lines, i, "PIPELINE")
            has_unroll = _check_nearby_pragma(lines, i, "UNROLL")
            body_lines = _count_body_lines(lines, i)
            nesting = len([d for d in loop_stack if d < current_depth])

            loops.append(LoopInfo(
                line_number=i + 1,
                loop_type="while",
                iteration_count=None,
                body_line_count=body_lines,
                has_pipeline_pragma=has_pipeline,
                has_unroll_pragma=has_unroll,
                nesting_depth=nesting,
            ))
            loop_stack.append(current_depth)

    return loops


def _check_nearby_pragma(lines: List[str], line_idx: int, pragma_type: str) -> bool:
    """Check if a pragma of given type exists within 2 lines before or after."""
    start = max(0, line_idx - 2)
    end = min(len(lines), line_idx + 3)
    for i in range(start, end):
        if f"#PRAGMA HLS {pragma_type}" in lines[i].upper():
            return True
    return False


def _count_body_lines(lines: List[str], start_idx: int) -> int:
    """Count lines in the body of a loop/function starting at start_idx."""
    depth = 0
    started = False
    count = 0
    for i in range(start_idx, min(start_idx + 200, len(lines))):
        line = lines[i]
        depth += line.count('{') - line.count('}')
        if '{' in line and not started:
            started = True
            continue
        if started:
            count += 1
            if depth <= 0:
                break
    return max(0, count - 1)  # Exclude closing brace


def _extract_arrays(lines: List[str]) -> List[ArrayInfo]:
    """Extract array declarations with dimensions."""
    arrays = []
    text = "\n".join(lines)

    for m in RE_ARRAY_DECL.finditer(text):
        element_type = m.group(1).strip()
        name = m.group(2)
        dims_str = m.group(3)

        # Parse dimensions [N][M] -> [N, M]
        dims = [int(d) for d in re.findall(r'\[(\d+)\]', dims_str)]

        # Find line number
        line_offset = text[:m.start()].count('\n')

        # Check for partition pragma nearby
        has_partition = False
        start = max(0, line_offset - 2)
        end = min(len(lines), line_offset + 3)
        for i in range(start, end):
            if "ARRAY_PARTITION" in lines[i].upper() and name in lines[i]:
                has_partition = True
                break

        arrays.append(ArrayInfo(
            line_number=line_offset + 1,
            name=name,
            element_type=element_type,
            dimensions=dims,
            has_partition_pragma=has_partition,
        ))

    return arrays


def _extract_pragmas(lines: List[str]) -> List[PragmaInfo]:
    """Extract HLS pragmas with categorization."""
    pragmas = []
    for i, line in enumerate(lines):
        m = RE_PRAGMA_HLS.match(line.strip())
        if m:
            directive_type = m.group(1).upper()
            rest = m.group(2).strip()
            full_text = f"#pragma HLS {m.group(1)}{' ' + rest if rest else ''}"

            category_map = {
                "PIPELINE": "pipeline",
                "UNROLL": "unroll",
                "ARRAY_PARTITION": "array_partition",
                "INTERFACE": "interface",
                "INLINE": "inline",
                "DEPENDENCE": "dependence",
                "RESOURCE": "resource",
                "STREAM": "stream",
                "DATAFLOW": "dataflow",
                "LOOP_MERGE": "loop_merge",
            }
            category = category_map.get(directive_type, "other")

            pragmas.append(PragmaInfo(
                line_number=i + 1,
                directive=full_text,
                category=category,
            ))

    return pragmas


def _extract_functions(lines: List[str]) -> List[FunctionInfo]:
    """Extract function definitions and basic call graph."""
    functions = []
    text = "\n".join(lines)

    # Find all function definitions
    func_names = set()
    func_locations = []
    for m in RE_FUNC_DEF.finditer(text):
        name = m.group(1)
        line_num = text[:m.start()].count('\n') + 1
        func_names.add(name)
        func_locations.append((name, line_num, m.start()))

    # For each function, find calls within its body
    for idx, (name, line_num, start_pos) in enumerate(func_locations):
        # Find body extent (next function or end of file)
        if idx + 1 < len(func_locations):
            end_pos = func_locations[idx + 1][2]
        else:
            end_pos = len(text)

        body = text[start_pos:end_pos]
        calls = set()
        for cm in RE_FUNC_CALL.finditer(body):
            called = cm.group(1)
            if called != name and called not in _BUILTIN_NAMES and called in func_names:
                calls.add(called)

        # Heuristic: first function with array params is likely top-level
        func_line = lines[line_num - 1] if line_num <= len(lines) else ""
        is_top = idx == 0 or ("[" in func_line and "void" in func_line)

        functions.append(FunctionInfo(
            line_number=line_num,
            name=name,
            is_top_level=is_top and idx == len(func_locations) - 1,
            calls=sorted(calls),
        ))

    return functions


def _infer_memory_pattern(
    loops: List[LoopInfo],
    arrays: List[ArrayInfo],
) -> str:
    """Heuristic memory access pattern inference.

    sequential: loops iterate linearly (i, i+1, i+2...)
    strided:    loops with stride > 1 or 2D array access
    random:     bit-reverse or computed indices
    unknown:    can't determine
    """
    if not loops or not arrays:
        return "unknown"

    # Check if any loop has a known sequential iteration
    has_sequential = any(
        l.iteration_count is not None and l.iteration_count > 0
        for l in loops
    )

    # Check for 2D arrays (suggests strided access)
    has_2d = any(len(a.dimensions) > 1 for a in arrays)

    if has_2d:
        return "strided"
    elif has_sequential:
        return "sequential"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_source(file_path: str) -> CodeProfile:
    """Analyze a SystemC/C++ source file and return a CodeProfile.

    Raises FileNotFoundError if the file doesn't exist.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    loops = _extract_loops(lines)
    arrays = _extract_arrays(lines)
    pragmas = _extract_pragmas(lines)
    functions = _extract_functions(lines)
    memory_pattern = _infer_memory_pattern(loops, arrays)

    return CodeProfile(
        file_path=file_path,
        total_lines=len(lines),
        loops=loops,
        arrays=arrays,
        pragmas=pragmas,
        functions=functions,
        memory_access_pattern=memory_pattern,
    )
