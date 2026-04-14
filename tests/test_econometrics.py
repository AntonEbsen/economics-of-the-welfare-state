import pandas as pd

from analysis.regression_utils import adjust_pvalues


def test_adjust_pvalues():
    pvals = pd.Series([0.01, 0.04, 0.1, 0.8], index=["v1", "v2", "v3", "v4"])
    adjusted = adjust_pvalues(pvals, method="bonferroni")

    # Bonferroni: 0.01 * 4 = 0.04
    assert adjusted.loc[adjusted["Variable"] == "v1", "Corrected P-Value"].values[0] == 0.04
    # 0.04 * 4 = 0.16
    assert adjusted.loc[adjusted["Variable"] == "v2", "Corrected P-Value"].values[0] == 0.16
