# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:04:52 2025

@author: laycocla
"""

import numpy as np
import pandas as pd

def LevineTS(x, y, common_x, common_y, scores, w1):
    
    """
  Perform Levine True Score equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  scores: Array of score range to equate
  w1: Weight for group 1

  Returns:
  DataFrame of equated scores
  """
  
    # Means and SDs
    mean_cx = np.mean(common_x)
    mean_cy = np.mean(common_y)
    sd_cx = np.std(common_x, ddof=1)
    sd_cy = np.std(common_y, ddof=1)

    # Apply the same formula as OS for now – update later with true score logic
    slope = sd_cy / sd_cx
    intercept = mean_cy - slope * mean_cx

    scores = np.arange(score_min, score_max + 1)
    equated = intercept + slope * scores

    return pd.DataFrame({"score": scores, "equated": equated})