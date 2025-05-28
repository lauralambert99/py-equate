# -*- coding: utf-8 -*-
"""
Created on Wed May 14 07:06:55 2025

@author: laycocla
"""

from ..freq_tab import joint_distribution
from ..freq_tab import common_item_marginal
from ..freq_tab import conditional_distribution
from ..freq_tab import reweight_conditional_distribution


import pandas as pd
import numpy as np

def fe(x, y, common_x, common_y, scores, w1):
    """
  Perform Frequency Estimation equating.

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
  
  #Cumulative synthetic distributions
    Fsx = fsx.cumsum()
    Gsy = gsy.cumsum()
  
    Psx = 100*(Fsx.shift(1, fill_value = 0) + fsx/2)
  
  #Make G(y) * 100 value column - easier to reference this
    Gsy_100 = Gsy*100

  #Make a function to take each P(x) value and find the smallest Gy_100 value that is => to it

  #But what we really want is the corresponding Y value

    def find_Y_star(Psx, Gsy, Y):
      idx = np.searchsorted(Gsy, Psx, side="left")
      return Y[idx] if idx < len(Y) else None

    scores = scores + 1
    pdata = pd.DataFrame({
      'Score': scores, 
      'Y': common_y, 
      'fsx': fsx.values, 
      'gsy': gsy.values, 
      'Fsx': Fsx.values, 
      'Gsy': Gsy.values,
      'Psx': Psx.values,
      'Gsy_100': Gsy_100.values
  })

  #Use the function to make a new column 
    pdata['Y_star_u'] = pdata['Psx'].apply(lambda Psx: find_Y_star(Psx, pdata['Gsy_100'], pdata['Score']))

  #Compute G(Y*u)
    pdata['Gsy_star'] = pdata['Y_star_u'].apply(lambda y_star: Gsy[y_star] if pd.notna(y_star) else None)
  
  #Will also need a lag Y*u for equation
    pdata['Gsy_star_lag'] = pdata['Y_star_u'].apply(lambda y_star: Gsy[y_star - 1] if pd.notna(y_star) and y_star > min(common_y) else None)

  # Compute equated scores
    pdata['e_yx'] = ((((pdata['Psx'] / 100) - pdata['Gsy_star_lag']) / 
                    (pdata['Gsy_star'] - pdata['Gsy_star_lag'])) + 
                    (pdata['Y_star_u'] - 0.5))

    return {'yx': pdata['e_yx']}
