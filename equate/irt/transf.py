# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 08:50:28 2025

@author: laycocla
"""

import numpy as np
import pandas as pd

def transf(aJ, bJ, aI, bI, method = "mean_mean"):
    """
  Perform mean/mean scale transformation.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  scores: Array of score range to equate
  
  method: what scale transformation to use; select from "mean_mean", "mean_sigma", "S_L" (Stocking & Lord), or "Haebara"

  Returns:
  ????
  """
#First, need a and b common item parameters to do the transformation

    #Method validatation
    valid_methods = ["mean_mean", "mean_sigma", "S_L", "Haebara"]
    if method not in valid_methods:
        raise ValueError(f"Method '{method}' not supported. Choose from {valid_methods}")
        
    if method == "mean_mean":
        A = np.std(bJ)/np.std(bI)
        B = np.mean(bJ) - A*np.mean(bI)
    
    elif method == "mean_sigma":
        A = np.mean(aI)/np.mean(aJ)
        B = np.mean(bJ) - A*np.mean(bI)
    
    elif method == "S_L":
        do stuff
    
    elif method = "Haebara":
        do other stuff
    
    else:
        raise ValueError(f"Unsupported method: {method}")
  
    #To do rescaling:
    #Brain is not braining: put form I onto form J to get form J equivalents.....WHICH ONE ARE WE CHANGING???
    data = pd.DataFrame({
        'Item': items, 
        'aJ': aJ, 
        'bJ': bJ, 
        'cJ': cJ, 
        'aI': aI, 
        'bI': bI, 
        'cI': cJ,
    })
    
    data['aJ_transf'] = aI/A
    data['bJ_transf'] = A*bIj + B
    
    thetaJ = A*thetaI + B
    
    return data