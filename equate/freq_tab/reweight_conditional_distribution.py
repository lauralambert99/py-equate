# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:03:31 2025

@author: laycocla
"""
import pandas as pd

def reweight_conditional_distribution(
    conditional_probs: pd.DataFrame,
    other_marginals: pd.Series,
    marginal_row_label = "Marginal_Common",
    exclude_last_n = 2
) -> pd.DataFrame:
    """
    Computes expected score distribution by reweighting conditional probabilities 
    using marginal distribution from the other population.

    Parameters:
    - conditional_probs: DataFrame with conditional probabilities P(score | common).
    - other_marginals: Series with marginal distribution of common scores from other group.
    - marginal_row_label: Row label to skip in conditional_probs.
    - exclude_last_n: Number of trailing columns to exclude (e.g., totals/cumulatives).

    Returns:
    - DataFrame with reweighted expected distribution.
    """
    weighted_df = conditional_probs.copy()

    #Columns over which to apply weights (exclude totals)
    process_cols = weighted_df.columns[:-exclude_last_n] if exclude_last_n > 0 else weighted_df.columns

    for row in weighted_df.index:
        if row != marginal_row_label:
            for col in process_cols:
                marginal_value = other_marginals.get(col, pd.NA)
                if pd.notna(marginal_value):
                    weighted_df.at[row, col] = weighted_df.at[row, col] * marginal_value
                else:
                    weighted_df.at[row, col] = pd.NA

    return weighted_df
