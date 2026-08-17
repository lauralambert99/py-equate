    
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:07:55 2025

@author: Laura
"""

#Load in dependent packages
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools import add_constant
#%%

def equipercen(x = None, y = None, score_min = None, score_max = None, presmoothed_df = None, order = None):
    """
    A function to perform random groups equipercentile equating from two score vectors
    
    Can accept raw vectors x, y or presmoothed_dict from presmooth().
    
    Parameters
    ----------
    x : list or pd.Series
        Scores on the OLD form.
    y : list or pd.Series
        Scores on the NEW form.
    score_min : int
        Minimum possible score.
    score_max : int
        Maximum possible score.
    presmoothed_df : pd.DataFrame, optional
        Presmoothed frequencies with columns ['form', 'score', 'smoothed_freq', 'order'].
    order : int, optional
        Which order of presmoothing to use (required if presmoothed_df is provided).

    Returns
    -------
    equated_df : pd.DataFrame
        Columns: 'score' and 'equated'.
    moments_df : pd.DataFrame
        Columns: 'mean', 'sd', 'skew', 'kurt'.
    """

    def single_equip(xfreq, yfreq):
        
        scores = xfreq.index.to_numpy()
        
        # Convert to probabilities
        f_x = xfreq / xfreq.sum()
        g_y = yfreq / yfreq.sum()

        # Cumulative distributions
        Fx = f_x.cumsum()
        Gy = g_y.cumsum()
        Px = 100 * (Fx.shift(1, fill_value=0) + f_x / 2)
        Gy_100 = (Gy * 100).to_numpy()

        # Helper function for upper Y* index
        def find_Y_star(px_val, Gy_val):
            idx = np.searchsorted(Gy_val, px_val, side="left")
            return min(idx, len(Gy_val)-1)


        # Interpolated equated scores
        e_yx = []
        for px in Px.to_numpy():
            k = find_Y_star(px, Gy_100)        # positional index in arrays
            Ghi = Gy.iloc[k]                   # positional lookup
            Glo = Gy.iloc[k-1] if k > 0 else 0.0
            if (Ghi <= Glo) or np.isclose(Ghi, Glo):
                yhat = scores[k].astype(float)
            else:
                # Linear interpolation within the Y bin, with 0.5 continuity correction
                yhat = (scores[k] - 0.5) + ((px/100.0 - Glo) / (Ghi - Glo))
            e_yx.append(float(yhat))

        equated_df = pd.DataFrame({'score': scores, 'equated': e_yx})
        return equated_df

    #Prepare frequencies
    if presmoothed_df is not None:
        if order is None:
            raise ValueError("Must specify 'order' when using presmoothed_df")
        xfreq = presmoothed_df.query(f"form=='X' & order=={order}")['smoothed_freq']
        yfreq = presmoothed_df.query(f"form=='Y' & order=={order}")['smoothed_freq']
        xfreq.index = range(score_min, score_max + 1)
        yfreq.index = range(score_min, score_max + 1)
    else:
        # Raw vectors -> frequency tables
        xfreq = x.value_counts().reindex(range(score_min, score_max + 1), fill_value=0).sort_index()
        yfreq = y.value_counts().reindex(range(score_min, score_max + 1), fill_value=0).sort_index()

    equated_df = single_equip(xfreq, yfreq)
    return equated_df
    
    

    