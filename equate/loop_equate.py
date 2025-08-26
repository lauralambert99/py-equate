# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:20:42 2025

@author: laycocla
"""

import pandas as pd
from .equipercen import equipercen

def loop_equate(presmoothed_df, score_min, score_max):
    """
    Perform equipercentile equating for all orders in a presmoothed DataFrame,
    returning a single DataFrame with one column per order.

    Parameters
    ----------
    presmoothed_df : pd.DataFrame
        Output of presmooth(), must have columns ['order', 'form', 'score', 'smoothed_freq']
    score_min : int
        Minimum possible score
    score_max : int
        Maximum possible score

    Returns
    -------
    equated_df : pd.DataFrame
        Columns: 'score', 'order_1', 'order_2', ..., 'order_n'
    """
    orders = sorted(presmoothed_df['order'].unique())
    df_records = pd.DataFrame({'score': range(score_min, score_max + 1)})

    for order in orders:
        eq_df = equipercen(
            x = None,
            y = None,
            score_min = score_min,
            score_max = score_max,
            presmoothed_df = presmoothed_df,
            order = order
        )
        df_records[f'order_{order}'] = eq_df['equated'].values

    return df_records
