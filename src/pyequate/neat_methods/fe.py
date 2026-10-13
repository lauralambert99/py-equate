# -*- coding: utf-8 -*-
"""
Created on Wed May 14 07:06:55 2025

@author: laycocla
"""

from ..freq_tab.common_item_marginal import common_item_marginal
from ..freq_tab.conditional_distribution import conditional_distribution
from ..freq_tab.reweight_conditional_distribution import reweight_conditional_distribution
from ..helper_fxns.equip_map import equip_map
from ..freq_tab.joint_distribution import _check_score_index


import pandas as pd
import numpy as np

def fe(gx, gy, score_min, score_max, w1):
    """
  Perform Frequency Estimation equating.

  Parameters:
  gx : pandas.DataFrame
        A joint distribution table for Form X and its common items, as produced by
        `joint_distribution()`. Must include columns for the unique scores on X
        and the corresponding frequencies.
    
    gy : pandas.DataFrame
        A joint distribution table for Form Y and its common items, as produced by
        `joint_distribution()`. Must include columns for the unique scores on Y
        and the corresponding frequencies.
        
  score_min: minimum possible score
  score_max: maximum possible score
  
  w1: Weight for group 1

  Returns
    -------
    pandas.DataFrame
        A score correspondence table with equivalent scores on Form X and Form Y,
        derived via frequency estimation equating.

    Notes
    -----
    This function assumes that the joint distributions `gx` and `gy` have already
    been computed. To construct these objects, use:

        gx = joint_distribution(df, form_col, common_col, ...)
        gy = joint_distribution(df, form_col, common_col, ...)

    where `df` is a data frame containing raw score data, `form_col` is the column name for total score, and `common_col` is the column name for the common (or anchor) items

"""
    _check_score_index(gx, score_min, score_max, 'gx')
    _check_score_index(gy, score_min, score_max, 'gy')
    
    #Define weights
    w2 = (1 - w1)
    
    #Define scores
    scores = np.arange(score_min, score_max + 1)

  
  #First, get joint distributions for each population, marginal distributions, and a cumulative distribution
    g1x_v2 = common_item_marginal(gx)
  
    g2y_v2 = common_item_marginal(gy)
  
  #Then, make conditional distribution tables
    cond_x = conditional_distribution(g1x_v2)
    cond_y = conditional_distribution(g2y_v2)
  
  #Calculate the opposite distributions for the forms
  #i.e., distribution of Form Y in population 1
    f2x = reweight_conditional_distribution(cond_x, other_marginals=g2y_v2.iloc[-1, :-2])
    g1y = reweight_conditional_distribution(cond_y, other_marginals=g1x_v2.iloc[-1, :-2])
    
  #Calculate synthetic population values
    f1x = gx['Marginal']
    g2y = gy['Marginal']
    
    f2x['Marginal'] = f2x.iloc[:, :-2].sum(axis=1)
    g1y['Marginal'] = g1y.iloc[:, :-2].sum(axis=1)

    f2x_2 = f2x['Marginal'].iloc[:-1]
    g1y_2 = g1y['Marginal'].iloc[:-1]

    f1x = gx['Marginal']
    g2y = gy['Marginal']

    f2x_2 /= f2x_2.sum()
    g1y_2 /= g1y_2.sum()

  
  #Marginal synthetic distributions
    fsx = w1*f1x + w2*f2x_2
    gsy = w1*g1y_2 + w2*g2y
  
  #Equipercentile equating of the synthetic distributions
    fsx = pd.Series(fsx.to_numpy(dtype=float), index=scores)
    gsy = pd.Series(gsy.to_numpy(dtype=float), index=scores)
    e_yx = equip_map(fsx, gsy)

    return pd.DataFrame({'Score': scores, 'e_yx': e_yx})
