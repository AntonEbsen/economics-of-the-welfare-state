"""
Tests for ``analysis.robustness.run_feedback_regressions`` — the
reverse-causality check that regresses a globalisation index on lagged
welfare-state transfers and lagged controls. Lifted from notebook
cell 67.

We build a synthetic panel where ``KOFGI = f(ln_gdppc_{t-1}) + noise``
and ``sstran`` is engineered to be independent of ``KOFGI``. The
feedback coefficient on ``sstran_lag1`` should therefore be close to
zero — and, more importantly, the helper should produce a well-formed
``linearmodels`` result object per index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.robustness import (
    export_feedback_regression_table,
    run_feedback_regressions,
)


def _synthetic_panel(seed: int = 11) -> pd.DataFrame:
    """12 countries × 25 years with all the columns run_feedback_regressions needs."""
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(12)]
    years = list(range(1995, 2020))

    alpha_g = {c: rng.normal(0, 1) for c in countries}
    alpha_s = {c: rng.normal(0, 1) for c in countries}

    rows = []
    for c in countries:
        for t in years:
            ln_gdppc = rng.normal(10, 0.5)
            inflation_cpi = rng.normal(2, 1)
            deficit = rng.normal(-2, 2)
            debt = rng.normal(50, 10)
            ln_population = rng.normal(16, 1)
            dependency_ratio = rng.normal(50, 5)
            # Globalization driven by gdp + country FE + noise, NOT by sstran.
            kofgi = alpha_g[c] + 0.5 * ln_gdppc + rng.normal(0, 0.3)
            # sstran driven by independent shocks.
            sstran = alpha_s[c] + rng.normal(0, 0.3)
            rows.append(
                {
                    "iso3": c,
                    "year": t,
                    "sstran": sstran,
                    "KOFGI": kofgi,
                    "KOFEcGI": kofgi + rng.normal(0, 0.2),
                    "KOFSoGI": kofgi + rng.normal(0, 0.2),
                    "KOFPoGI": kofgi + rng.normal(0, 0.2),
                    "ln_gdppc": ln_gdppc,
                    "inflation_cpi": inflation_cpi,
                    "deficit": deficit,
                    "debt": debt,
                    "ln_population": ln_population,
                    "dependency_ratio": dependency_ratio,
                }
            )
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "indices": ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"],
        "controls": [
            "ln_gdppc",
            "inflation_cpi",
            "deficit",
            "debt",
            "ln_population",
            "dependency_ratio",
        ],
    }


def test_run_feedback_regressions_returns_one_model_per_index():
    df = _synthetic_panel()
    config = _config()
    models = run_feedback_regressions(df, config)
    assert set(models.keys()) == set(config["indices"])
    # Each value should be a linearmodels PanelResults — probe via params attr.
    for name, result in models.items():
        assert hasattr(result, "params"), f"{name}: expected PanelResults, got {type(result)}"
        assert "sstran_lag1" in result.params.index


def test_run_feedback_regressions_respects_indices_arg():
    df = _synthetic_panel()
    models = run_feedback_regressions(df, _config(), indices=["KOFGI"])
    assert set(models.keys()) == {"KOFGI"}


def test_export_feedback_regression_table_writes_latex(tmp_path):
    df = _synthetic_panel()
    out_path = export_feedback_regression_table(df, _config(), out_dir=tmp_path)
    assert out_path.name == "feedback_regression_table.tex"
    assert out_path.exists() and out_path.stat().st_size > 0
    text = out_path.read_text(encoding="utf-8")
    # LATEX_LABEL_MAP should have rewritten sstran_lag1 to a presentation label.
    # We check a weaker invariant: the file is non-trivial LaTeX.
    assert "\\begin" in text
