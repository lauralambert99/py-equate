# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:30:15 2025

@author: Laura
"""

#TODO: question - do we want full possible range of scores, or just reported scores?

import numpy as np
import pandas as pd

from ..helper_fxns.equip_map import equip_map

def chained(x, y, common_x, common_y, score_min, score_max, anchor_min, anchor_max, type = "linear"):
    """
  Perform chained equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  
  score_min: minimum possible score
  score_max: maximum possible score
  
  anchor_min: minimum possible anchor score
  anchor_max: maximum possible anchor score
  
  type: of chained equating; options are "linear" and "eq" (equipercentile).  Default is "linear".
  

  Returns:
  DataFrame of equated scores
  """
    #Define scores
    scores = np.arange(score_min, score_max + 1)
    
    if type == "linear":  
         
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
    
    elif type == "eq":
        x, y = pd.Series(x), pd.Series(y)
        common_x, common_y = pd.Series(common_x), pd.Series(common_y)

        score_pts = range(score_min, score_max + 1)
        anchor_pts = range(anchor_min, anchor_max + 1)

        #Frequency tables, keeping score points nobody obtained
        f_x = x.value_counts().reindex(score_pts, fill_value=0).sort_index()
        h1_v = common_x.value_counts().reindex(anchor_pts, fill_value=0).sort_index()
        h2_v = common_y.value_counts().reindex(anchor_pts, fill_value=0).sort_index()
        g_y = y.value_counts().reindex(score_pts, fill_value=0).sort_index()

        #Form X to common items in Population 1: e_V1(x)
        e_vx = equip_map(f_x, h1_v)

        #Common items to Form Y in Population 2: e_Y2(v)
        e_yv = equip_map(h2_v, g_y)

        #Chain them: e_Y(x) = e_Y2(e_V1(x)); e_V1(x) is non-integer, so interpolate e_Y2
        e_yx = np.interp(e_vx, np.asarray(anchor_pts, dtype=float), e_yv)

        return pd.DataFrame({'Score': scores,
                             'e_yx': e_yx})

    else:
        raise ValueError(f"Unsupported type: {type}")