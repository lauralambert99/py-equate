# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:00:41 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .irt_helper import ts_curve 

def irtTS(formX_params, formY_params, score_range=None, model='2pl', D=1.7, A=1.0, B=0.0):
    """
    Perform IRT True Score Equating
    
    This function transforms Form X parameters to the Form Y scale, NOT vice versa.
    This matches the R implementation in SNSequate::irt.eq
    
    Parameters:
    - formX_params: DataFrame with item parameters ('a', 'b', 'c') for Form X
    - formY_params: DataFrame with item parameters ('a', 'b', 'c') for Form Y
    - score_range: Iterable of observed score values on Form X
    - model: IRT model ('1pl', '2pl', or '3pl')
    - D: Scaling constant (default 1.7)
    - A: Scale linking parameter (default 1.0)
    - B: Scale linking parameter (default 0.0)
    
    Returns:
    - DataFrame with columns: 'X', 'Theta', 'tyx'

    """
    formX_params = formX_params.copy()
    formY_params = formY_params.copy()
    
    #Do transformation on Form X parameters
    formX_params['a'] = formX_params['a'] / A
    formX_params['b'] = formX_params['b'] * A + B
    
    if 'c' not in formX_params.columns:
        formX_params['c'] = 0.0
    if 'c' not in formY_params.columns:
        formY_params['c'] = 0.0
    
    #Compute true score at a given theta for Form X (transformed)
    def true_score_X_at_theta(theta):
        """Compute true score for Form X at a single theta value"""
        return ts_curve(formX_params, np.array([theta]), model=model, D=D)[0]
    
    #Compute true score at a given theta for Form Y
    def true_score_Y_at_theta(theta):
        """Compute true score for Form Y at the same theta"""
        return ts_curve(formY_params, np.array([theta]), model=model, D=D)[0]
    
    #Determine score range
    if score_range is None:
        score_max = len(formY_params)
        score_range = np.arange(0, score_max + 1)
    
    #For each observed score, find theta using root finding
    tyx = []
    thetas = []
    
    for x in score_range:
        #Find theta where true_score_X(theta) = x
        try:
            #Function to find root of: true_score_X(theta) - x = 0
            theta_x = brentq(lambda t: true_score_X_at_theta(t) - x, -40, 40)
        except ValueError:
            #If root not found in range, handle edge cases
            ts_at_minus40 = true_score_X_at_theta(-40)
            ts_at_plus40 = true_score_X_at_theta(40)
            
            if x <= ts_at_minus40:
                theta_x = -40
            elif x >= ts_at_plus40:
                theta_x = 40
            else:
                # This shouldn't happen, but just in case
                print(f"Warning: Could not find theta for score {x}")
                theta_x = np.nan
        
        #Compute Form Y true score at that same theta
        y = true_score_Y_at_theta(theta_x)
        
        thetas.append(theta_x)
        tyx.append(y)
    
    out = pd.DataFrame({
        "X": score_range,
        "Theta": thetas,
        "tyx": tyx
    })
    
    return out