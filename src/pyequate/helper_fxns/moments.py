# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:16:22 2025

@author: laycocla
"""

import pandas as pd
from scipy.stats import skew, kurtosis

def moments(eq_df, score_col = 'score'):
    """
    Compute moments (mean, SD, skew, kurtosis) for each equated score column.

    Parameters
    ----------
    eq_df : pd.DataFrame
        equating output, with at least one column of equated scores.
    
    score: column in the dataframe for raw scores (this will be skipped)

    Returns
    -------
    moments_df : pd.DataFrame
        Columns: ['order', 'mean', 'sd', 'skew', 'kurt']
    """
    
    moments_records = []
    for col in eq_df.columns:
        if col == score_col:
            continue
        moments_records.append({
            'column': col,
            'mean': eq_df[col].mean(),
            'sd': eq_df[col].std(),
            'skew': skew(eq_df[col]),
            'kurt': kurtosis(eq_df[col])
        })
    return pd.DataFrame(moments_records)
