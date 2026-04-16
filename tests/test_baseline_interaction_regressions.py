"""
Tests for the baseline and interaction regression helpers lifted from
``notebooks/02_modern_pipeline.ipynb`` cells 54 (baseline) and 60/61
(interaction).

We build a synthetic panel where ``sstran`` is driven by ``KOFGI_{t-1}``
and a small amount of noise so each PanelOLS result has a valid
parameter vector and a reasonably well-behaved coefficient. The tests
lock the public contract (output keys, required LaTeX filenames,
``ValueError`` when no index columns are present) rather than the
numeric stability of the coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.robustness import (
    export_baseline_regression_table,
    export_interaction_excl_postcommunist_table,
    export_interaction_regression_table,
    export_marginal_effects_tables,
    run_baseline_regressions,
    run_interaction_regressions,
    run_interaction_regressions_excl_postcommunist,
)


def _synthetic_regime_panel(seed: int = 33) -> pd.DataFrame:
    """12 countries × 25 years with the four indices + regime dummies."""
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(12)]
    years = list(range(1995, 2020))

    # Assign each country to a regime so the interaction terms have non-zero
    # variance. All five regimes rotate across the 12 synthetic countries —
    # social-democratic is the reference category so its interaction column
    # is intentionally omitted; we still need a few countries in it to avoid
    # perfect multicollinearity with the base lagged index.
    regime_cycle = [
        "conservative",
        "mediterranean",
        "liberal",
        "post_communist",
        "social_democratic",
    ]
    country_regime = {c: regime_cycle[i % len(regime_cycle)] for i, c in enumerate(countries)}

    alpha = {c: rng.normal(0, 1) for c in countries}
    rows = []
    for c in countries:
        for t in years:
            kofgi = 60.0 + rng.normal(0, 5)
            sstran = alpha[c] + 0.1 * kofgi + rng.normal(0, 0.5)
            row = {
                "iso3": c,
                "year": t,
                "sstran": sstran,
                "KOFGI": kofgi,
                "KOFEcGI": kofgi + rng.normal(0, 2),
                "KOFSoGI": kofgi + rng.normal(0, 2),
                "KOFPoGI": kofgi + rng.normal(0, 2),
                "ln_gdppc": rng.normal(10, 0.5),
                "inflation_cpi": rng.normal(2, 1),
                "deficit": rng.normal(-2, 2),
                "debt": rng.normal(50, 10),
                "ln_population": rng.normal(16, 1),
                "dependency_ratio": rng.normal(50, 5),
                # Regime dummies required by `prepare_regression_data(interactions=True)`
                "regime_conservative": 0,
                "regime_mediterranean": 0,
                "regime_liberal": 0,
                "regime_post_communist": 0,
                "regime_social_democratic": 0,
            }
            row[f"regime_{country_regime[c]}"] = 1
            rows.append(row)
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
        "dependent_var": "sstran",
    }


# ---------------------------------------------------------------------------
# Baseline regressions
# ---------------------------------------------------------------------------


def test_run_baseline_regressions_returns_one_model_per_index():
    df = _synthetic_regime_panel()
    models = run_baseline_regressions(df, _config())
    assert set(models.keys()) == {"KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"}
    for name, result in models.items():
        assert hasattr(result, "params"), f"{name}: expected PanelResults"
        assert f"{name}_lag1" in result.params.index


def test_run_baseline_regressions_respects_indices_arg():
    df = _synthetic_regime_panel()
    models = run_baseline_regressions(df, _config(), indices=["KOFGI"])
    assert set(models.keys()) == {"KOFGI"}


def test_run_baseline_regressions_skips_missing_index_columns():
    df = _synthetic_regime_panel().drop(columns=["KOFPoGI"])
    models = run_baseline_regressions(df, _config())
    assert "KOFPoGI" not in models
    assert "KOFGI" in models


def test_export_baseline_regression_table_writes_latex(tmp_path):
    df = _synthetic_regime_panel()
    out_path = export_baseline_regression_table(df, _config(), out_dir=tmp_path)
    assert out_path.name == "baseline_regression_table.tex"
    assert out_path.exists() and out_path.stat().st_size > 0
    assert "\\begin" in out_path.read_text(encoding="utf-8")


def test_export_baseline_regression_table_raises_when_no_indices(tmp_path):
    df = _synthetic_regime_panel().drop(columns=["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    with pytest.raises(ValueError, match="No baseline models"):
        export_baseline_regression_table(df, _config(), out_dir=tmp_path)


# ---------------------------------------------------------------------------
# Interaction regressions (regime-heterogeneity)
# ---------------------------------------------------------------------------


def test_run_interaction_regressions_includes_regime_interaction_terms():
    df = _synthetic_regime_panel()
    models = run_interaction_regressions(df, _config())
    assert set(models.keys()) == {"KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"}
    # The interaction terms ride on top of the base lagged index.
    kofgi_params = models["KOFGI"].params.index
    for term in (
        "KOFGI_lag1",
        "int_conservative",
        "int_mediterranean",
        "int_liberal",
        "int_post_communist",
    ):
        assert term in kofgi_params, f"{term} missing from KOFGI interaction model"


def test_export_interaction_regression_table_writes_latex(tmp_path):
    df = _synthetic_regime_panel()
    out_path = export_interaction_regression_table(df, _config(), out_dir=tmp_path)
    assert out_path.name == "interaction_regression_table.tex"
    assert out_path.exists() and out_path.stat().st_size > 0
    assert "\\begin" in out_path.read_text(encoding="utf-8")


def test_export_interaction_regression_table_raises_when_no_indices(tmp_path):
    df = _synthetic_regime_panel().drop(columns=["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    with pytest.raises(ValueError, match="No interaction models"):
        export_interaction_regression_table(df, _config(), out_dir=tmp_path)


# ---------------------------------------------------------------------------
# Marginal-effects tables
# ---------------------------------------------------------------------------


def test_export_marginal_effects_tables_writes_one_file_per_index(tmp_path):
    df = _synthetic_regime_panel()
    paths = export_marginal_effects_tables(df, _config(), out_dir=tmp_path)
    assert set(paths.keys()) == {"KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"}
    for idx_name, out_path in paths.items():
        assert out_path.name == f"marginal_effects_{idx_name}.tex"
        assert out_path.exists() and out_path.stat().st_size > 0
        text = out_path.read_text(encoding="utf-8")
        # Each regime row-label from generate_marginal_effects should survive.
        for regime in ("Social Democrat", "Conservative", "Mediterranean", "Liberal"):
            assert regime in text, f"{idx_name}: missing regime row {regime!r}"


def test_export_marginal_effects_tables_respects_indices_arg(tmp_path):
    df = _synthetic_regime_panel()
    paths = export_marginal_effects_tables(df, _config(), out_dir=tmp_path, indices=["KOFGI"])
    assert set(paths.keys()) == {"KOFGI"}


def test_export_marginal_effects_tables_raises_when_no_indices(tmp_path):
    df = _synthetic_regime_panel().drop(columns=["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    with pytest.raises(ValueError, match="No interaction models"):
        export_marginal_effects_tables(df, _config(), out_dir=tmp_path)


# ---------------------------------------------------------------------------
# Post-communist exclusion robustness (notebook cell 59)
# ---------------------------------------------------------------------------


def test_run_excl_postcommunist_has_three_interaction_terms():
    df = _synthetic_regime_panel()
    models = run_interaction_regressions_excl_postcommunist(df, _config())
    assert len(models) > 0
    # Pick any model — it should have 3 interaction terms, NOT 4
    result = next(iter(models.values()))
    params = result.params.index
    for term in ("int_conservative", "int_mediterranean", "int_liberal"):
        assert term in params, f"{term} missing from excl-postcommunist model"
    assert "int_post_communist" not in params


def test_run_excl_postcommunist_skips_missing_index():
    df = _synthetic_regime_panel().drop(columns=["KOFPoGI"])
    models = run_interaction_regressions_excl_postcommunist(df, _config())
    assert "KOFPoGI" not in models and len(models) == 3


def test_export_excl_postcommunist_table_writes_latex(tmp_path):
    df = _synthetic_regime_panel()
    out_path = export_interaction_excl_postcommunist_table(df, _config(), out_dir=tmp_path)
    assert out_path.name == "interaction_excl_postcommunist_table.tex"
    assert out_path.exists() and out_path.stat().st_size > 0
    text = out_path.read_text(encoding="utf-8")
    assert "\\begin" in text
    # Should NOT contain the post-communist interaction term
    assert "int\\_post\\_communist" not in text or "int_post_communist" not in text


def test_export_excl_postcommunist_table_raises_when_no_indices(tmp_path):
    df = _synthetic_regime_panel().drop(columns=["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    with pytest.raises(ValueError, match="No interaction.*excl.*post-communist"):
        export_interaction_excl_postcommunist_table(df, _config(), out_dir=tmp_path)
