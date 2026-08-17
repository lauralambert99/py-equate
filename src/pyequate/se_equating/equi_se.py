# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 05:32:50 2025

@author: laycocla
"""
import numpy as np
import pandas as pd

def eq_see_asy(scores_x, freq_X, scores_y, freq_Y):
    """
    Calculate Standard Errors of Equating (SEE) for equipercentile equating
    using asymptotic SEE formula.
    
    Parameters
    ----------
    freq_X : pd.Series
        Frequencies for scores on form X, indexed by score (ascending).
        Must be the same length as scores_X.
    freq_Y : pd.Series
        Frequencies for scores on form Y, indexed by score (ascending).
        Must be the same length as scores_Y.
    scores_X : array-like
        Unique score points for form X (i.e., possible score range)
    scores_Y : array-like
        Unique score points for form Y
        
    Returns
    -------
    pd.DataFrame with columns:
        - score : score on form X
        - equated : equated score on form Y
        - SEE : standard error of equating
    """

    scores_x = np.array(scores_x, dtype=float)
    freq_X = np.array(freq_X, dtype=float)
    scores_y = np.array(scores_y, dtype=float)
    freq_Y = np.array(freq_Y, dtype=float)
    
    #Make sure things are the same length
    if len(scores_x) != len(freq_X):
        raise ValueError(f"Length mismatch: scores_x ({len(scores_x)}) != freq_X ({len(freq_X)}).")
    if len(scores_y) != len(freq_Y):
        raise ValueError(f"Length mismatch: scores_y ({len(scores_y)}) != freq_Y ({len(freq_Y)}).")
    
    #Sample sizes
    nX = np.sum(freq_X) 
    nY = np.sum(freq_Y)
        
    #Probabilities
    f_x = freq_X / nX
    g_y = freq_Y / nY
    
    #Cumulative distributions
    Fx = np.cumsum(f_x)
    Gy = np.cumsum(g_y)
    
    #Percentiles
    Px = np.zeros(len(scores_x))
    
    for i in range(len(scores_x)):
        if i == 0:
            Px[i] = f_x[i] / 2.0
        else:
            Px[i] = Fx[i-1] + f_x[i] / 2.0
    
    Px = Px * 100
    Gy_100 = Gy * 100
    
    def find_Y_star(px_val, Gy_val):
        idx = np.searchsorted(Gy_val, px_val, side="left")
        return min(idx, len(Gy_val)-1)

    e_yx = []
    
    #This is the equating bit
    for i, px in enumerate(Px):
        k = find_Y_star(px, Gy_100)
        Ghi = Gy[k]
        Glo = Gy[k-1] if k > 0 else 0.0
        
        if (Ghi <= Glo) or np.isclose(Ghi, Glo):
            yhat = scores_y[k].astype(float)
        else:
            yhat = (scores_y[k] - 0.5) + ((px/100.0 - Glo) / (Ghi - Glo))
            
        e_yx.append(float(yhat))
        
        #This is the SEE bit
        se = np.zeros(len(scores_x))
        
        for i in range(len(scores_x)):
            px_val = Px[i] / 100.0
            
            #Find which Y scores bracket this percentile
            k = np.searchsorted(Gy_100, Px[i], side="left")
            k = min(k, len(Gy_100) - 1)
            
            #Cumulative probabilities
            Ghi = Gy[k]  
            Glo = Gy[k-1] if k > 0 else 0.0 
            
            #Density interval: G(y*) - G(y*-1)
            g0 = Ghi - Glo
            
            #Don't divide by zero!
            if g0 < 1e-10:
                g0 = 1e-10
            
            #SE = sqrt((1-q)*q/(n_x*g0^2) + (1/(n_y*g0^2))*(gm - q^2 + ((q-gm)^2)/g0)) - this is how equate calculated it; used a density interval
            #gm = Glo
            
            #Separate the terms to make math easier
            term1 = (1 - px_val) * px_val / (nX * g0**2)
            term2 = (1 / (nY * g0**2)) * (Glo - px_val**2 + ((px_val - Glo)**2) / g0)
            var_eY = term1 + term2
            
            se[i] = np.sqrt(var_eY)

    equated_df = pd.DataFrame({
        'score': scores_x,
        'equated': e_yx,
        'SEE': se
    })
    
    return equated_df
    
    