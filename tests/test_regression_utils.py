"""
Unit tests for src/analysis/regression_utils.py covering the fitting functions.

Complements tests/test_analysis.py (which tests the lighter-weight
``prepare_regression_data`` and ``generate_marginal_effects`` helpers).

These tests build small synthetic panels where the population DGP is known,
so we can check that PanelOLS recovers the true slope within tolerance and
that downstream helpers (Hausman, event study, placebo) produce output in the
documented shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.regression_utils import (
    LATEX_LABEL_MAP,
    prepare_regression_data,
    run_event_study,
    run_hausman_test,
    run_panel_ols,
    run_placebo_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_panel(
    n_countries: int = 6,
    n_years: int = 20,
    true_beta: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesize a panel with a known slope: y = alpha_i + lambda_t + beta*x + eps.

    Returned in long format with a MultiIndex ``(iso3, year)`` suitable for
    PanelOLS. Also includes a ``ctrl`` column and welfare-regime dummies so
    the same fixture can drive interaction and control tests.
    """
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(n_countries)]
    years = list(range(2000, 2000 + n_years))

    alpha = {c: rng.normal(0, 1) for c in countries}
    lam = {t: rng.normal(0, 0.5) for t in years}

    rows = []
    for c in countries:
        for t in years:
            x = rng.normal(0, 1)
            ctrl = rng.normal(0, 1)
            eps = rng.normal(0, 0.2)
            y = alpha[c] + lam[t] + true_beta * x + 0.1 * ctrl + eps
            rows.append({"iso3": c, "year": t, "y": y, "x": x, "ctrl": ctrl})

    df = pd.DataFrame(rows)
    # Assign welfare regimes deterministically so interaction tests have coverage.
    df["regime_conservative"] = (df["iso3"].isin(countries[:2])).astype(int)
    df["regime_mediterranean"] = (df["iso3"].isin(countries[2:3])).astype(int)
    df["regime_liberal"] = (df["iso3"].isin(countries[3:4])).astype(int)
    df["regime_post_communist"] = (df["iso3"].isin(countries[4:5])).astype(int)
    return df


@pytest.fixture
def panel_df():
    return _make_panel()


@pytest.fixture
def panel_indexed(panel_df):
    """Same fixture with the (iso3, year) MultiIndex set."""
    return panel_df.set_index(["iso3", "year"])


# ---------------------------------------------------------------------------
# run_panel_ols
# ---------------------------------------------------------------------------


def test_run_panel_ols_recovers_true_slope(panel_indexed):
    """With a known DGP the estimated beta should be within tolerance."""
    results = run_panel_ols(panel_indexed, "y", ["x", "ctrl"])
    assert "x" in results.params.index
    assert results.params["x"] == pytest.approx(0.5, abs=0.1)


def test_run_panel_ols_produces_standard_outputs(panel_indexed):
    """Two-way FE fit should produce the usual PanelOLS result surface."""
    results = run_panel_ols(panel_indexed, "y", ["x"])
    # Standard linearmodels output attributes.
    assert hasattr(results, "params")
    assert hasattr(results, "std_errors")
    assert hasattr(results, "rsquared")
    assert "x" in results.params.index
    # Two-way FE absorbs the constant, and time dummies should not surface
    # as individual params on the exog matrix.
    assert results.params.shape[0] >= 1


def test_run_panel_ols_supports_unclustered_cov(panel_indexed):
    """cov_type='unadjusted' bypasses the clustered branch."""
    results = run_panel_ols(panel_indexed, "y", ["x"], cov_type="unadjusted")
    assert "x" in results.params.index


# ---------------------------------------------------------------------------
# run_hausman_test
# ---------------------------------------------------------------------------


def test_run_hausman_test_returns_expected_shape(panel_indexed):
    out = run_hausman_test(panel_indexed, "y", ["x", "ctrl"])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "Hausman Statistic",
        "Degrees of Freedom",
        "P-Value",
        "Verdict (α=0.05)",
    ]
    assert len(out) == 1
    assert 0.0 <= out["P-Value"].iloc[0] <= 1.0
    # DF is the number of shared non-constant coefficients: x and ctrl = 2.
    assert out["Degrees of Freedom"].iloc[0] == 2


def test_run_hausman_verdict_text_is_one_of_two_branches(panel_indexed):
    verdict = run_hausman_test(panel_indexed, "y", ["x"])["Verdict (α=0.05)"].iloc[0]
    assert verdict.startswith("Reject H") or verdict.startswith("Fail to reject")


# ---------------------------------------------------------------------------
# run_event_study
# ---------------------------------------------------------------------------


def test_run_event_study_builds_window_and_drops_baseline(panel_indexed):
    plot_df, _res = run_event_study(panel_indexed, "y", treat_var="x", event_year=2010, window=3)
    # window=3 means rel_time in [-3, 3] → 7 rows total (baseline rel_time=-1 is zero-coef).
    assert len(plot_df) == 7
    assert set(plot_df.columns) == {"rel_time", "coef", "lower", "upper"}
    baseline = plot_df[plot_df["rel_time"] == -1].iloc[0]
    # The baseline period is pinned to zero in the constructor.
    assert baseline["coef"] == 0.0
    assert baseline["lower"] == 0.0
    assert baseline["upper"] == 0.0


def test_run_event_study_confidence_intervals_are_ordered(panel_indexed):
    plot_df, _ = run_event_study(panel_indexed, "y", treat_var="x", event_year=2010, window=3)
    non_baseline = plot_df[plot_df["rel_time"] != -1]
    assert (non_baseline["lower"] <= non_baseline["coef"]).all()
    assert (non_baseline["coef"] <= non_baseline["upper"]).all()


# ---------------------------------------------------------------------------
# run_placebo_test
# ---------------------------------------------------------------------------


def test_run_placebo_test_returns_distribution_centered_near_zero(panel_indexed):
    """Shuffling within entity should destroy the true signal."""
    coefs = run_placebo_test(
        panel_indexed, dep_var="y", indep_var="x", exog_vars=["x", "ctrl"], n_sims=20
    )
    assert isinstance(coefs, np.ndarray)
    assert coefs.shape == (20,)
    # True beta is 0.5, placebo should be close to 0.
    assert abs(np.mean(coefs)) < 0.3


# ---------------------------------------------------------------------------
# Interaction / prepare_regression_data + run_panel_ols integration
# ---------------------------------------------------------------------------


def test_interaction_regression_runs_end_to_end(panel_df):
    """prepare_regression_data(interactions=True) + run_panel_ols should fit."""
    ols_data, exog = prepare_regression_data(
        panel_df, dep_var="y", indep_var="x", ctrls_lagged=["ctrl"], interactions=True
    )
    assert "int_conservative" in exog
    # Some interactions will be collinear with entity FE but the fit should
    # still produce coefficients for the non-collinear subset — we just
    # assert we can fit without errors.
    results = run_panel_ols(ols_data, "y", exog)
    assert "x" in results.params.index


# ---------------------------------------------------------------------------
# LATEX_LABEL_MAP sanity
# ---------------------------------------------------------------------------


def test_latex_label_map_covers_core_variables():
    """All variables documented in the README should have LaTeX labels."""
    for key in (
        "KOFGI\\_lag1",
        "ln\\_gdppc\\_lag1",
        "inflation\\_cpi\\_lag1",
        "deficit\\_lag1",
        "debt\\_lag1",
        "ln\\_population\\_lag1",
        "dependency\\_ratio\\_lag1",
        "const",
    ):
        assert key in LATEX_LABEL_MAP, f"Missing LaTeX label for {key}"
