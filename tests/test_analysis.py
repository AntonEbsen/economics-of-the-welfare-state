import numpy as np
import pandas as pd

from analysis.regression_utils import generate_marginal_effects, prepare_regression_data


def test_prepare_regression_data():
    # Create dummy data
    data = {
        "iso3": ["USA", "USA", "CAN", "CAN"],
        "year": [2000, 2001, 2000, 2001],
        "y": [1, 2, 3, 4],
        "x": [5, 6, 7, 8],
        "ctrl": [9, 10, 11, 12],
        "regime_conservative": [0, 0, 1, 1],
        "regime_mediterranean": [0, 0, 0, 0],
        "regime_liberal": [1, 1, 0, 0],
        "regime_post_communist": [0, 0, 0, 0],
    }
    df = pd.DataFrame(data)

    # Test without interactions
    ols_data, exog = prepare_regression_data(df, "y", "x", ["ctrl"], interactions=False)
    assert "x" in exog
    assert "ctrl" in exog
    assert "int_conservative" not in exog
    assert isinstance(ols_data.index, pd.MultiIndex)

    # Test with interactions
    ols_data, exog = prepare_regression_data(df, "y", "x", ["ctrl"], interactions=True)
    assert "int_conservative" in exog
    assert ols_data["int_conservative"].iloc[2] == 7 * 1  # x * regime_conservative for CAN


def test_generate_marginal_effects():
    # Mock results object with params, covariance matrix, and df_resid
    class MockResults:
        def __init__(self, params, cov, df_resid=100):
            self.params = params
            self.cov = cov
            self.df_resid = df_resid

    var_names = ["x", "int_conservative", "int_mediterranean", "int_liberal", "int_post_communist"]
    params = pd.Series(
        [0.5, 0.2, -0.1, 0.0, 0.3],
        index=var_names,
    )
    # Simple diagonal covariance (SE = 0.1 for all)
    cov = pd.DataFrame(
        np.eye(5) * 0.01,
        index=var_names,
        columns=var_names,
    )
    results = MockResults(params, cov)

    me_table = generate_marginal_effects(results, "x")

    assert (
        me_table.loc[
            me_table["Welfare Regime"] == "Social Democrat (Ref)", "Marginal Effect"
        ].values[0]
        == 0.5
    )
    assert (
        me_table.loc[me_table["Welfare Regime"] == "Conservative", "Marginal Effect"].values[0]
        == 0.7
    )  # 0.5 + 0.2
    assert (
        me_table.loc[me_table["Welfare Regime"] == "Mediterranean", "Marginal Effect"].values[0]
        == 0.4
    )  # 0.5 - 0.1

    # New columns from the SE upgrade
    assert "Std. Error" in me_table.columns
    assert "p-value" in me_table.columns
    assert "Sig." in me_table.columns
    # Reference group SE should be sqrt(0.01) = 0.1
    ref_se = me_table.loc[
        me_table["Welfare Regime"] == "Social Democrat (Ref)", "Std. Error"
    ].values[0]
    assert abs(ref_se - 0.1) < 1e-10
