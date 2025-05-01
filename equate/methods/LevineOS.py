# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:05:02 2025

@author: laycocla
"""

import numpy as np
import pandas as pd

def LevineOS(x, y, common_x, common_y, scores, w1):
    """
  Perform Levine Observed Score equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  scores: Array of score range to equate
  w1: Weight for group 1

  Returns:
  DataFrame of equated scores
  """
    
    #Calculate Mean and SD of common items
    mean_cx = np.mean(common_x)
    mean_cy = np.mean(common_y)
    
    sd_cx = np.std(common_x, ddof=1)
    sd_cy = np.std(common_y, ddof=1)

    #Synthetic population slope and intercept
    slope = sd_cy / sd_cx
    intercept = mean_cy - slope * mean_cx
    
    #Equate
    ly_x = intercept + slope * scores

    eyx = pd.DataFrame({'Scores': scores,
                       'ey': ly_x})
    return eyx
