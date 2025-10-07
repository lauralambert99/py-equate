# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:19:35 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools import add_constant

def presmooth(x, y, score_min, score_max, max_order = 10):
    """
    Perform loglinear presmoothing for X and Y using polynomial orders 1..max_order.
    
    Parameters
    ----------
    freq_x: array-like of observed frequencies for X
    freq_y: array-like of observed frequencies for Y
    max_order: int, maximum polynomial degree to try
    
    Returns
    -------
    presmooth_df : pd.DataFrame
        Columns: ['order', 'form', 'score', 'smoothed_freq']
        Each row corresponds to a presmoothed frequency for a particular score and order
    """
    x = pd.Series(x).reindex(range(score_min, score_max + 1), fill_value=0)
    y = pd.Series(y).reindex(range(score_min, score_max + 1), fill_value=0)
    
    all_records = []

    def smooth_freq(freq_series, form_label):
        for order in range(1, max_order + 1):
            #Construct matrix for log-linear Poisson regression
            scores = np.arange(score_min, score_max + 1)
            X_design = np.vstack([scores**i for i in range(1, order + 1)]).T
            X_design = add_constant(X_design, has_constant='add')
            model = Poisson(freq_series.values, X_design)
            fit = model.fit(disp=0)
            smoothed = np.exp(fit.predict(X_design))
            
            for s, val in zip(scores, smoothed):
                all_records.append({
                    'order': order,
                    'form': form_label,
                    'score': s,
                    'smoothed_freq': val
                })

    smooth_freq(x, 'X')
    smooth_freq(y, 'Y')

    presmooth_df = pd.DataFrame(all_records)
    return presmooth_df