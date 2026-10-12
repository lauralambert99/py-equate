# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:00:35 2025

@author: laycocla
"""

#Discussion: how to best define these so it's clear which form is which?  

import pandas as pd

#This gets the joint distributions of uncommon and common items
#Also calculates marginal and cumulative distributions for the uncommon items
#Need to add marginal distibution for common items as a row
def joint_distribution(df, form_col, common_col, form_range = None, common_range = None):
    """
    Create a joint probability distribution of form score vs common-item score, 
    including marginal and cumulative probabilities.
    
    Parameters:
        df (pd.DataFrame): DataFrame with score columns
        form_col (str): Column name for form score
        common_col (str): Column name for common-item score
        form_range (list or None): Optional list of form score values to include
        common_range (list or None): Optional list of common score values to include
        
    Returns:
        pd.DataFrame: Joint distribution with marginal and cumulative probabilities

    Raises:
        ValueError: if any score falls outside form_range or common_range
    """
    
    form_scores = df[form_col]
    common_scores = df[common_col]

    #Determine score ranges from data if not provided
    if form_range is None:
        form_range = range(int(form_scores.min()), int(form_scores.max()) + 1)

    if common_range is None:
        common_range = range(int(common_scores.min()), int(common_scores.max()) + 1)

    #Refuse scores outside the requested ranges rather than silently reshaping the table
    out_x = sorted(set(form_scores.dropna()) - set(form_range))
    out_v = sorted(set(common_scores.dropna()) - set(common_range))
    if out_x:
        raise ValueError(f"Scores in '{form_col}' outside form_range: {out_x}")
    if out_v:
        raise ValueError(f"Scores in '{common_col}' outside common_range: {out_v}")

    #Joint frequency table, including score points nobody obtained
    joint = pd.crosstab(form_scores, common_scores).reindex(
        index=list(form_range), columns=list(common_range), fill_value=0)
    joint.index.name = None
    joint.columns.name = None

    #Normalize to probabilities
    total = joint.to_numpy().sum()
    joint_prob = joint / total

    #Marginal and cumulative distributions
    joint_prob['Marginal'] = joint_prob.sum(axis = 1)
    joint_prob['Cumulative'] = joint_prob['Marginal'].cumsum()

    return joint_prob
