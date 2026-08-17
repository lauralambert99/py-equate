# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:00:35 2025

@author: laycocla
"""

#Discussion: how to best define these so it's clear which form is which?  

import pandas as pd
import numpy as np

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
    """
    
    #Determine score ranges from data if not provided
    if form_range is None:
        form_range = range(df[form_col].min(), df[form_col].max() + 1)

    if common_range is None:
        common_range = range(df[common_col].min(), df[common_col].max() + 1)


    #Make an empty joint frequency table
    joint = pd.DataFrame(0, index = form_range, columns = common_range)
    
    #Count joint occurrences
    for _, row in df.iterrows():
        x = row[form_col]
        v = row[common_col]
        if x in joint.index and v in joint.columns:
            joint.at[x, v] += 1
        else:
            #Optionally add missing score to table instead of skipping
            joint = joint.reindex(index=sorted(set(joint.index).union({x})))
            joint = joint.reindex(columns=sorted(set(joint.columns).union({v})))
            joint.at[x, v] = 1  #Initialize to 1 (was 0, now first seen)

    #Normalize to probabilities
    total = joint.to_numpy().sum()
    joint_prob = joint / total

    #Marginal and cumulative distributions
    joint_prob['Marginal'] = joint_prob.sum(axis = 1)
    joint_prob['Cumulative'] = joint_prob['Marginal'].cumsum()

    return joint_prob
