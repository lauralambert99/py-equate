# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 08:50:28 2025

@author: laycocla
"""

import numpy as np
import pandas as pd

#TODO: I bet we can require a dataframe with specified columns to make this neater.

def transf(aJ, bJ, cJ, aI, bI, items, common, method = "mean_mean"):
    """
  Perform mean/mean scale transformation.

  Parameters:
  aJ, bJ, cJ: Array of item parameters for the old form
  aI, bI: Array of item parameters for the new form
  common: Array of common items (e.g., 1:5)
  items: Array of item numbers (e.g., 1, 2, 3, etc.)
  
  method: what scale transformation to use; select from "mean_mean", "mean_sigma", "S_L" (Stocking & Lord), or "Haebara"

  Returns:
  A dataframe with item parameters on the old and new forms, and the transformed parameters.
  """
#First, need a and b common item parameters to do the transformation

    #Method validatation
    valid_methods = ["mean_mean", "mean_sigma", "S_L", "Haebara"]
    if method not in valid_methods:
        raise ValueError(f"Method '{method}' not supported. Choose from {valid_methods}")
    
    data = pd.DataFrame({
        'Item': items, 
        'aJ': aJ, 
        'bJ': bJ, 
        'cJ': cJ, 
        'aI': aI, 
        'bI': bI, 
        'cI': cJ,
    })
    
    #TODO: specify only do these things on common items!
    
    if method == "mean_mean":
        A = np.std(bJ)/np.std(bI)
        B = np.mean(bJ) - A*np.mean(bI)
    
    elif method == "mean_sigma":
        A = np.mean(aI)/np.mean(aJ)
        B = np.mean(bJ) - A*np.mean(bI)
    
    #elif method == "S_L":
        #do stuff iteratively using Gauss-Hermite quadrature to approximate the integral in the minimized function
    
    #elif method = "Haebara":
       #do other stuff iteratively
    
    else:
        raise ValueError(f"Unsupported method: {method}")
  
    #To do rescaling:
    #Brain is not braining: put form I onto form J to get form J equivalents.....WHICH ONE ARE WE CHANGING???

    
    data['aJ_transf'] = aI/A
    data['bJ_transf'] = A*bI + B
    
    thetaJ = A*thetaI + B
    
    return data