"""
Guard-rail tests for ``notebooks/02_modern_pipeline.ipynb``.

After lifting the heavy cells into ``src/analysis/`` modules (Tier 5.22),
the notebook relies on those modules via bare ``from analysis.X import
Y`` statements. If a maintainer renames or deletes one of the extracted
helpers, the notebook silently breaks — it's not exercised by the
regular pytest suite.

These tests parse the notebook JSON, extract every top-level import
statement from every code cell, and verify each symbol still resolves.
No cell execution is attempted (no matplotlib display, no data IO);
we rely on plain ``importlib`` so the check is fast and dependency-
free beyond the modules the notebook already imports.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_modern_pipeline.ipynb"


def _extract_imports(source: str) -> list[tuple[str, str]]:
    """Return ``[(module, symbol)]`` pairs from the top-level imports in ``source``.

    Handles both ``from X import a, b`` and ``import X`` forms. Uses
    :mod:`ast` so we ignore imports that live inside function bodies or
    conditional branches — only module-level statements matter for the
    drift-detection invariant we care about.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Skip malformed cells silently; not the job of this test to flag them.
        return []

    pairs: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                pairs.append((node.module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                pairs.append((alias.name, ""))
    return pairs


def _notebook_imports() -> list[tuple[str, str, int]]:
    """Collect ``(module, symbol, cell_index)`` from every code cell."""
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    imports: list[tuple[str, str, int]] = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for module, symbol in _extract_imports(source):
            imports.append((module, symbol, i))
    return imports


def test_notebook_exists():
    assert NOTEBOOK_PATH.exists(), f"Notebook missing at {NOTEBOOK_PATH}"


def test_all_notebook_imports_resolve():
    """Every ``from X import Y`` at the notebook top level must resolve.

    We import each module once (catches missing modules), then verify
    each symbol is an attribute (catches renames).
    """
    pairs = _notebook_imports()
    assert pairs, "No imports parsed from notebook — parser regression?"

    failures: list[str] = []
    for module, symbol, cell_idx in pairs:
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover — failure path is the point
            failures.append(f"cell {cell_idx}: cannot import {module!r} ({exc})")
            continue
        if symbol and not hasattr(mod, symbol):
            failures.append(f"cell {cell_idx}: {module}.{symbol} not found")

    if failures:
        pytest.fail(
            "Notebook imports no longer resolve — library refactor drift:\n  "
            + "\n  ".join(failures)
        )


def test_thin_cells_call_expected_helpers():
    """Lock the thin-call contract for the seven headline cells.

    If a future maintainer re-inlines one of the extracted cells (e.g.
    by copy-pasting the function body back into the notebook), this
    test catches it: the marker string must appear at least once
    somewhere in the notebook.
    """
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        nb_text = fh.read()
    expected_markers = [
        "plot_sstran_trend",
        "plot_kof_trend",
        "export_subcomponent_regression_table",
        "export_correlation_matrix",
        "export_baseline_regression_table",
        "export_interaction_regression_table",
        "export_interaction_excl_postcommunist_table",
        "export_feedback_regression_table",
        "export_marginal_effects_tables",
    ]
    missing = [m for m in expected_markers if m not in nb_text]
    assert not missing, f"Expected thin-call markers missing: {missing}"
