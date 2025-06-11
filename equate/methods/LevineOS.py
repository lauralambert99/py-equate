# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:05:02 2025

@author: laycocla
"""


import numpy as np
import pandas as pd

def LevineOS(x, y, common_x, common_y, scores, w1, anchor = "internal"):
    """
  Perform Levine Observed Score equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  scores: Array of score range to equate
  w1: Weight for group 1
  anchor: If anchor items are "internal" or "external".  Defaults to "internal".

  Returns:
  DataFrame of equated scores
  """
  
    #Method validatation
    valid_anchor = ["internal", "external"]
    if anchor not in valid_anchor:
        raise ValueError(f"Anchor '{anchor}' not supported. Choose from {valid_anchor}")
   
    #Define weights
    w2 = (1 - w1)
    
    if anchor == "internal":
        gamma_1 = np.var(x)/np.cov(x, common_x)
        gamma_2 = np.var(y)/np.cov(y, common_y)
    
    elif anchor == "external":
        gamma_1 = (np.var(x) + np.cov(x, common_x))/(np.var(common_x) + np.cov(x, common_x))
        gamma_2 = (np.var(y) + np.cov(y, common_y))/(np.var(common_y) + np.cov(y, common_y))
    
    else:
        raise ValueError(f"Unsupported anchor selection: {anchor}")
    
    #Synthetic population stuff
    mu_sx = np.mean(x) - w2*gamma_1*(np.mean(common_x) - np.mean(common_y))
    mu_sy = np.mean(y) + w1*gamma_2*(np.mean(common_x) - np.mean(common_y))

    var_sx = (x.var() - w2*(gamma_1**2)*(common_x.var() - common_y.var()) + w1*w2*(gamma_1**2)*(np.mean(common_x) - np.mean(common_y))**2)
    var_sy = (y.var() + w1*(gamma_2**2)*(common_x.var() - common_y.var()) + w1*w2*(gamma_2**2)*(np.mean(common_x) - np.mean(common_y))**2)

    #Get standard deviations
    sd_sx = np.sqrt(var_sx)
    sd_sy = np.sqrt(var_sy)
    
    ly_x = (sd_sx/sd_sy)*(scores - mu_sx) + mu_sy

    eyx = pd.DataFrame({'Scores': scores,
                       'ey': ly_x})
    return eyx

