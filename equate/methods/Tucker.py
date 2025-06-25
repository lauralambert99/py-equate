# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:30:41 2025

@author: laycocla
"""
import pandas as pd
import numpy as np

def Tucker(x, y, common_x, common_y, scores, w1):
    """
  Perform Tucker equating.

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
    
    #Calculate gamma
    gamma_1 = common_x.cov(x)/common_x.var(ddof=1)
    gamma_2 = common_y.cov(y)/common_y.var(ddof=1)

    #Synthetic population stuff
    mu_sx = np.mean(x) - w2*gamma_1*(np.mean(common_x) - np.mean(common_y))
    mu_sy = np.mean(y) + w1*gamma_2*(np.mean(common_x) - np.mean(common_y))

    var_sx = (x.var(ddof=1) - w2*(gamma_1**2)*(common_x.var(ddof=1) - common_y.var(ddof=1)) + w1*w2*(gamma_1**2)*(np.mean(common_x) - np.mean(common_y))**2)
    var_sy = (y.var(ddof=1) + w1*(gamma_2**2)*(common_x.var(ddof=1) - common_y.var(ddof=1)) + w1*w2*(gamma_2**2)*(np.mean(common_x) - np.mean(common_y))**2)

    #Get standard deviations
    sd_sx = np.sqrt(var_sx)
    sd_sy = np.sqrt(var_sy)
    
    ly_x = (sd_sx/sd_sy)*(scores - mu_sx) + mu_sy

    eyx = pd.DataFrame({'Scores': scores,
                       'ey': ly_x})
    return eyx
