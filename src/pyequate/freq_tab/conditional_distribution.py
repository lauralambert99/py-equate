# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:00:36 2025

@author: laycocla
"""
import pandas as pd

def conditional_distribution(joint_probs: pd.DataFrame, marginal_row_label="Marginal_Common", exclude_last_n=2) -> pd.DataFrame:
    """
    Computes the conditional distribution of total scores given common-item scores
    by dividing each column by the marginal distribution of the common-item scores.

    Parameters:
    - joint_probs: DataFrame with joint probabilities including a row labeled `marginal_row_label`.
    - marginal_row_label: Name of the row containing marginal distribution for the common items.
    If feeding straigh in, will be "Marginal_Common"
    - exclude_last_n: Number of columns at the end to exclude from the conditional computation.

    Returns:
    - DataFrame of the same shape, with conditional probabilities replacing the joint ones,
      and NaNs in excluded columns and the marginal row.
    """
    cond_probs = joint_probs.copy()

    #Columns to divide over
    process_cols = cond_probs.columns[:-exclude_last_n] if exclude_last_n > 0 else cond_probs.columns

    #Extract marginal row values for divisor
    marginals = cond_probs.loc[marginal_row_label, process_cols]

    #Apply division to all rows EXCEPT the marginal row - don't want it messed up
    for row in cond_probs.index:
        if row != marginal_row_label:
            for col in process_cols:
                denom = marginals[col]
                cond_probs.at[row, col] = cond_probs.at[row, col] / denom if denom != 0 else pd.NA

    return cond_probs