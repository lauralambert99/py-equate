# -*- coding: utf-8 -*-
"""
Created on Wed May 14 07:06:59 2025

@author: laycocla
"""

from ..freq_tab.common_item_marginal import common_item_marginal
from ..freq_tab.conditional_distribution import conditional_distribution
from ..freq_tab.reweight_conditional_distribution import reweight_conditional_distribution


import pandas as pd
import numpy as np

def bh(x, y, gx, gy, score_min, score_max, w1):
    """
  Perform Braun-Holland equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  
  gx : pandas.DataFrame
        A joint distribution table for Form X and its common items, as produced by
        `joint_distribution()`. Must include columns for the unique scores on X
        and the corresponding frequencies.
    
    gy : pandas.DataFrame
        A joint distribution table for Form Y and its common items, as produced by
        `joint_distribution()`. Must include columns for the unique scores on Y
        and the corresponding frequencies.
        
  scores: Array of score range to equate
  
  w1: Weight for group 1

  Returns:
  DataFrame of equated scores
  """
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
    cond_x_pop2 = reweight_conditional_distribution(cond_x, other_marginals = cond_y.iloc[-1])
    cond_y_pop1 = reweight_conditional_distribution(cond_y, other_marginals = cond_x.iloc[-1])
  
  #Calculate synthetic population values
    f1x = gx['Marginal']
    f2x = cond_x_pop2.iloc[:-1]['Marginal']

    g1y = cond_y_pop1.iloc[:-1]['Marginal']
    g2y = gy['Marginal']

  
  #Marginal synthetic distributions
    fsx = w1*f1x + w2*f2x
    gsy = w1*g1y + w2*g2y
    
    x = x.value_counts().reindex(scores, fill_value=0).sort_index()
    y = y.value_counts().reindex(scores, fill_value=0).sort_index()  
    
    #Calculate means and standard deviations

    mu_sx = (scores * fsx).sum()
    var_sx = (((scores - mu_sx) ** 2) * fsx).sum()
    sd_sx = var_sx ** 0.5

    mu_sy = (scores * gsy).sum()
    var_sy = (((scores - mu_sy) ** 2) * gsy).sum()
    sd_sy = var_sy ** 0.5

    if sd_sx == 0:
        raise ValueError("sd_sx is zero (no spread in synthetic X). Cannot compute slope.")

    slope = sd_sy / sd_sx
    intercept = mu_sy - slope * mu_sx

    ex = slope * scores + intercept

    eq = pd.DataFrame({'Score': scores, 'ex': ex})
    return eq
    