# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:42:38 2025

@author: laycocla
"""

def test_irtTS_basic():
    from equate.irt.irtTS import irtTS
    import pandas as pd

    #Minimal test case: 2 items per form, 2PL
    dfX = pd.DataFrame({'a': [1, 1.2], 'b': [0, 1], 'c': [0, 0]})
    dfY = pd.DataFrame({'a': [0.8, 1.1], 'b': [-0.5, 1.5], 'c': [0, 0]})

    result = irtTS(dfX, dfY, score_range = range(0, 3), model = '2pl')

    assert result.shape[0] == 3
    assert "X" in result.columns and "tyx" in result.columns
    assert result["tyx"].iloc[0] < result["tyx"].iloc[2]
