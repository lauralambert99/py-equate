# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:23:30 2025

@author: Laura
"""
import numpy as np
import pandas as pd
import pytest
from equate.methods.Tucker import Tucker

@pytest.fixture
def mock_data():
    np.random.seed(123)
    x = np.random.normal(loc=25, scale=5, size=100)
    y = np.random.normal(loc=27, scale=5, size=100)
    common_x = np.random.normal(loc=15, scale=3, size=100)
    common_y = np.random.normal(loc=15.5, scale=3, size=100)
    scores = np.arange(0, 41)
    return x, y, common_x, common_y, scores

def test_tucker_output(mock_data):
    x, y, common_x, common_y, scores = mock_data
    result = Tucker(x, y, common_x, common_y, scores, w1=0.5)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Scores", "ey"]
    assert len(result) == len(scores)
    assert np.isfinite(result["ey"]).all()