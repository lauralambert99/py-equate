# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:00:36 2025

@author: laycocla
"""
import pandas as pd

def common_item_marginal(joint_probs: pd.DataFrame, exclude_last_n: int = 2) -> pd.DataFrame:
    """
    Adds a new row with the marginal distribution of common-item scores
    by summing over rows, excluding the last `exclude_last_n` columns.
    
    Using this function immediately following the joint_distribution function
    creates an initial common-item and Form distribution for a population
    
    Parameters:
    - joint_probs: DataFrame containing the joint probabilities.
    Feed the joint_distribution DataFrame straight to this
    
    - exclude_last_n: How many columns to exclude from the marginal row. (2)
    
    Returns:
    - Modified DataFrame with a new marginal row added.
    """
    # Identify which columns to include in the marginal
    include_cols = joint_probs.columns[:-exclude_last_n] if exclude_last_n > 0 else joint_probs.columns
    marginal_row = joint_probs[include_cols].sum(axis = 0)

    # Fill the excluded columns with NaNs to maintain shape
    for col in joint_probs.columns[-exclude_last_n:]:
        marginal_row[col] = pd.NA

    # Append the marginal row with a label
    marginal_row.name = "Marginal_Common"
    return pd.concat([joint_probs, pd.DataFrame([marginal_row])])
