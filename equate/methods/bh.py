# -*- coding: utf-8 -*-
"""
Created on Wed May 14 07:06:59 2025

@author: laycocla
"""

from ..freq_tab import joint_distribution
from ..freq_tab import common_item_marginal
from ..freq_tab import conditional_distribution
from ..freq_tab import reweight_conditional_distribution


import pandas as pd
import numpy as np

def bh(x, y, common_x, common_y, scores, w1):
    """
  Perform Braun-Holland equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  scores: Array of score range to equate
  w1: Weight for group 1

  Returns:
  DataFrame of equated scores
  """
  #Define weights
    w2 = (1 - w1)
  
  #First, get joint distributions for each population, marginal distributions, and a cumulative distribution
    g1x_v = joint_distribution(x, common_x)
    g1x_v = common_item_marginal(g1x_v)
  
    g2y_v = joint_distribution(y, common_y)
    g2y_v = common_item_marginal(g2y_v)
  
  #Then, make conditional distribution tables
    cond_x = conditional_distribution(g1x_v)
    cond_y = conditional_distribution(g2y_v)
  
  #Calculate the opposite distributions for the forms
  #i.e., distribution of Form Y in population 1
    cond_x_pop2 = reweight_conditional_distribution(cond_x, other_marginals = cond_y.iloc[-1])
    cond_y_pop1 = reweight_conditional_distribution(cond_y, other_marginals = cond_x.iloc[-1])
  
  #Calculate synthetic population values
    f1x = g1x_v['Marginal']
    f2x = cond_x_pop2.iloc[:-1]['Marginal']

    g1y = cond_y_pop1.iloc[:-1]['Marginal']
    g2y = g2y_v['Marginal']

  
  #Marginal synthetic distributions
    fsx = w1*f1x + w2*f2x
    gsy = w1*g1y + w2*g2y
    
    #Calculate means and standard deviations
    mu_sx = sum(x*fsx)  #Mean of synthetic population
    var_sx = sum(((x-mu_sx)**2)*fsx)  #Variance of synthtic population
    sd_sx = var_sx**0.5
    
    mu_sy = sum(y*gsy)  #Mean of synthetic population
    var_sy = sum(((x-mu_sy)**2)*gsy)  #Variance of synthtic population
    sd_sy = var_sy**0.5
    
    slope = sd_sy / sd_sx  #Ratio of standard deviations
    
    intercept = mu_sy - slope * mu_sx  
    
    ex = slope * scores + intercept
    
    eq =  pd.DataFrame({'Score': scores,
                        'ex': ex})
    #Return equated scores as a dataframe
    return eq
  