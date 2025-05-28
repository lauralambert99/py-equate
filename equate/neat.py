# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:00:57 2025

@author: Laura
"""

from .methods.Tucker import Tucker
from .methods.LevineOS import LevineOS
from .methods.LevineTS import LevineTS
from .methods.fe import fe
from .methods.bh import bh

import numpy as np
import pandas as pd

def neat(x, y, common_x, common_y, score_min, score_max, w1, items = "internal", method = "Tucker"):
    """
    Dispatches a single NEAT equating method.

    Parameters:
    - x, y: Array of raw scores for Form X and Form Y
    - common_x, common_y: Array of anchor scores
    - score_min, score_max: Score range of Form X to equate
    - w1: Weight for group 1 (0 < w1 < 1)
    - items: "internal" or "external" anchor design
    - method: NEAT equating method (options include "Tucker", "LevineOS" (Levine observed score), 
                                    "LevineTS" (Levine true score), "fe" (frequency estimation), 
                                    and "bh"(Braun-Holland))

    Returns:
    - DataFrame of equated scores
    """
    #TODO: Potential errors
        #For internal, common items score should not be larger than total score!

    #Weight validataion
    if not (0 <= w1 <= 1):
        raise ValueError("w1 must be between 0 and 1")
        
    #Method validatation
    valid_methods = ["Tucker", "LevineOS", "LevineTS", "fe", "bh"]
    if method not in valid_methods:
       raise ValueError(f"Method '{method}' not supported. Choose from {valid_methods}")
        
    #Define scores
    scores = np.arange(score_min, score_max + 1)

    if method == "Tucker":
        return Tucker(x, y, common_x, common_y, scores, w1)

    elif method == "LevineOS":
        return LevineOS(x, y, common_x, common_y, scores, w1)

    elif method == "LevineTS":
        return LevineTS(x, y, common_x, common_y, scores, w1)

    elif method == "fe":
        return fe(x, y, common_x, common_y, scores, w1)

    elif method == "bh":
        return bh(x, y, common_x, common_y, scores, w1)

    else:
        raise ValueError(f"Unsupported method: {method}")
    
    
