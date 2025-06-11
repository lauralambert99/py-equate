# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:30:15 2025

@author: Laura
"""

#TODO: question - do we want full possible range of scores, or just reported scores?

import numpy as np
import pandas as pd

def chained(x, y, common_x, common_y, score_min, score_max, type = "linear"):
    """
  Perform chained equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  
  score_min: minimum possible score
  score_max: maximum possible score
  
  type: of chained equating; options are "linear" and "eq" (equipercentile).  Default is "linear".
  

  Returns:
  DataFrame of equated scores
  """
    #Define scores
    scores = np.arange(score_min, score_max + 1)
    
    #Calculate gammas
    gamma_1 = np.std(x, ddof = 1)/np.std(common_x, ddof = 1)
    gamma_2 = np.std(y, ddof = 1)/np.std(common_y, ddof = 1)

    #Calculate Means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    mean_cx = np.mean(common_x)
    mean_cy = np.mean(common_y)
    
    
    #Chained equating equation
    lyx = (mean_y + gamma_2*(mean_cx - mean_cy) - (gamma_2/gamma_1)*mean_x) + (gamma_2/gamma_1)*scores
    
    eyx = pd.DataFrame({'Scores': scores,
                       'ey': lyx})
    return eyx