    
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
        # Convert to probabilities
        f_x = xfreq / xfreq.sum()
        g_y = yfreq / yfreq.sum()

        # Cumulative distributions
        Fx = f_x.cumsum()
        Gy = g_y.cumsum()
        Px = 100 * (Fx.shift(1, fill_value=0) + f_x / 2)
        Gy_100 = Gy * 100

        # Helper function for upper Y* index
        def find_Y_star(px_val, Gy_val, scores):
            idx = np.searchsorted(Gy_val, px_val, side="left")
            return scores[idx] if idx < len(scores) else scores[-1]

        scores = np.arange(len(xfreq))
        pdata = pd.DataFrame({
            'Score': scores,
            'f_x': f_x.values,
            'g_y': g_y.values,
            'Fx': Fx.values,
            'Gy': Gy.values,
            'Px': Px.values,
            'Gy_100': Gy_100.values
        })

        pdata['Y_star_u'] = pdata['Px'].apply(lambda px: find_Y_star(px, pdata['Gy_100'], scores))
        pdata['Gy_star'] = pdata['Y_star_u'].apply(lambda y_star: Gy[y_star] if pd.notna(y_star) else None)
        pdata['Gy_star_lag'] = pdata['Y_star_u'].apply(lambda y_star: Gy[y_star-1] if pd.notna(y_star) and y_star > 0 else None)

        # Interpolated equated scores
        e_yx = []
        for i, row in pdata.iterrows():
            Ghi = row['Gy_star']
            Glo = row['Gy_star_lag']
            Px_ = row['Px'] / 100
            Ystar = row['Y_star_u']

            if pd.isna(Ghi) or pd.isna(Glo) or Ghi == Glo:
                e_yx.append(float(Ystar))
            else:
                interp = ((Px_ - Glo) / (Ghi - Glo)) + (Ystar - 0.5)
                e_yx.append(float(interp))

        pdata['e_yx'] = e_yx

        equated_df = pd.DataFrame({
            'score': pdata['Score'],
            'equated': pdata['e_yx']
        })

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
    
    
#%%
equipercen(x = formx['x'], y = formy['x'], score_min = 0, score_max = 50, presmoothed_df = None, order = None)
    