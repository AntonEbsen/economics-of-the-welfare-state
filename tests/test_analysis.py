import pytest
import pandas as pd
import numpy as np
from analysis.regression_utils import prepare_regression_data, generate_marginal_effects

def test_prepare_regression_data():
    # Create dummy data
    data = {
        'iso3': ['USA', 'USA', 'CAN', 'CAN'],
        'year': [2000, 2001, 2000, 2001],
        'y': [1, 2, 3, 4],
        'x': [5, 6, 7, 8],
        'ctrl': [9, 10, 11, 12],
        'regime_conservative': [0, 0, 1, 1],
        'regime_mediterranean': [0, 0, 0, 0],
        'regime_liberal': [1, 1, 0, 0],
        'regime_post_communist': [0, 0, 0, 0],
    }
    df = pd.DataFrame(data)
    
    # Test without interactions
    ols_data, exog = prepare_regression_data(df, 'y', 'x', ['ctrl'], interactions=False)
    assert 'x' in exog
    assert 'ctrl' in exog
    assert 'int_conservative' not in exog
    assert isinstance(ols_data.index, pd.MultiIndex)
    
    # Test with interactions
    ols_data, exog = prepare_regression_data(df, 'y', 'x', ['ctrl'], interactions=True)
    assert 'int_conservative' in exog
    assert ols_data['int_conservative'].iloc[2] == 7 * 1 # x * regime_conservative for CAN

def test_generate_marginal_effects():
    # Mock results object class
    class MockResults:
        def __init__(self, params):
            self.params = params
    
    params = pd.Series({
        'x': 0.5,
        'int_conservative': 0.2,
        'int_mediterranean': -0.1,
        'int_liberal': 0.0,
        'int_post_communist': 0.3
    })
    results = MockResults(params)
    
    me_table = generate_marginal_effects(results, 'x')
    
    assert me_table.loc[me_table['Welfare Regime'] == 'Social Democrat (Ref)', 'Marginal Effect'].values[0] == 0.5
    assert me_table.loc[me_table['Welfare Regime'] == 'Conservative', 'Marginal Effect'].values[0] == 0.7 # 0.5 + 0.2
    assert me_table.loc[me_table['Welfare Regime'] == 'Mediterranean', 'Marginal Effect'].values[0] == 0.4 # 0.5 - 0.1
