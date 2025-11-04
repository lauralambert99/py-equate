# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 08:50:28 2025

@author: laycocla
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

#TODO: I bet we can require a dataframe with specified columns to make this neater.
#TODO: Report out what the A and B coefficients are

def transf(aJ, bJ, cJ, aI, bI, cI, items, common, method = "mean_mean"):
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
        'cI': cI,  
    })
    
    #Specify only do these things on common items
    common_data = data[data['Item'].isin(common)]
    
    if method == "mean_sigma":
        A = np.std(common_data['bJ']) / np.std(common_data['bI'])
        B = np.mean(common_data['bJ']) - A * np.mean(common_data['bI'])

    elif method == "mean_mean":
        A = np.mean(common_data['aI']) / np.mean(common_data['aJ'])
        B = np.mean(common_data['bJ']) - A * np.mean(common_data['bI'])
    
    #elif method == "S_L":
        #do stuff iteratively using Gauss-Hermite quadrature to approximate the integral in the minimized function
    
    elif method == "Haebara":
       #Haebara minimizes Hcrit = SUM[Hdiff(theta)]
       #Restrict to common items only
       aI = common_data['aI'].to_numpy()
       bI = common_data['bI'].to_numpy()
       cI = common_data['cI'].to_numpy()

       aJ = common_data['aJ'].to_numpy()
       bJ = common_data['bJ'].to_numpy()
       cJ = common_data['cJ'].to_numpy()
       
       #Need a theta grid
       theta = np.linspace(-4, 4, 30)
       
       #Also need a function for probability that 'examinees of a given ability will answer a particular item correctly'
       def theta_prob(theta, a, b, c):
            return c + (1 - c) * expit(a * (theta[:, None] - b)) #Expit is a neater way to do 1/(1-e^-x) i.e. logistic sigmoid fxn
       
       #Next, sum function
       def Haebara_sum(params):
           A, B = params #Tells minimize function to minimize these
           
           #Don't do anything to form J
           P_J = theta_prob(theta, aJ, bJ, cJ) #This makes a grid of all thetas
           
           #Make changes to form I
           a_transformed = aI/A
           b_transformed = A * bI + B
           
           P_I = theta_prob(theta, a_transformed, b_transformed, cI)
           
           #Sum the things
           return np.sum((P_I - P_J) ** 2) #Sum of the differences
       
       #Minimize Haebara
       result = minimize(Haebara_sum,
                         x0 = [1.0, 0.0],
                         method = 'L-BFGS-B',
                         bounds = [(0.1, 10.0), (-5.0, 5.0)])
       
       A, B = result.x
       
       #Message if it doesn't work
       if not result.success:
           raise RuntimeError(f"Haebara minimization failed: {sum_min.message}")
   
    else:
        raise ValueError(f"Unsupported method: {method}")
  
    #To do rescaling:
    #Brain is not braining: put form I onto form J to get form J equivalents.....WHICH ONE ARE WE CHANGING???

    
    data['aJ_transf'] = data['aI'] / A
    data['bJ_transf'] = A * data['bI'] + B
    
    #thetaJ = A*thetaI + B #Don't have thetas yet
    
    print("A =",A, "B =",B)
    return data

