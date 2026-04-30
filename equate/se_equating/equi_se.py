# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 05:32:50 2025

@author: laycocla
"""
import numpy as np
import pandas as pd

#Pull in equipercen() functions for now
from .equipercen import equipercen
from .equipercen import _freq_from_smoothed
from .equipercen import _freq_from_raw
from .equipercen import _find_Y_star
from .equipercen import _single_equip

def eq_see_asy(
        score_min: int, 
        score_max: int,
        x = None,
        y = None,
        x_smoothed = None,
        y_smoothed = None,
        x_degree = None,
        y_degree = None,) -> pd.DataFrame:
    """
    Calculate Standard Errors of Equating (SEE) for equipercentile equating, 
    using asymptotic SEE formula.
    
    Parameters
    ----------
    Parameters
    ----------
    score_min: int
        Minimum possible score on both forms.
    score_max: int
        Maximum possible score on both forms.
    x: array-like, optional
        Raw score vector for Form X (the old form being equated FROM).
    y: array-like, optional
        Raw score vector for Form Y (the new form being equated TO).
    x_smoothed: pd.Series | pd.DataFrame | dict, optional
        Direct output of loglinear() for Form X.
    y_smoothed: pd.Series | pd.DataFrame | dict, optional
        Direct output of loglinear() for Form Y.
    x_degree: int or str, optional
        Degree to select when x_loglinear is a stepup DataFrame,
        e.g. 4 or "Degree 4". Ignored for Series / choose-dict inputs.
    y_degree: int or str, optional
        Same as x_degree but for Form Y.
        
    Returns
    -------
    pd.DataFrame with columns:
        - score : score on form X
        - equated : equated score on form Y
        - SEE : standard error of equating
    """

    # ------------------------------------------------------------------
    # Equipercentile Equating
    # ------------------------------------------------------------------
    equated_df = equipercen(
        score_min = score_min,
        score_max = score_max,
        x = x,
        y = y,
        x_smoothed = x_smoothed,
        y_smoothed = y_smoothed,
        x_degree = x_degree,
        y_degree = y_degree,)
    
    # ------------------------------------------------------------------
    # Make frequency bits for SEE for smoothed scores
    # ------------------------------------------------------------------
    if x_smoothed is not None:
        xfreq = _freq_from_smoothed(x_smoothed, score_min, score_max, x_degree)
    else:
        xfreq = _freq_from_raw(x, score_min, score_max)

    if y_smoothed is not None:
        yfreq = _freq_from_smoothed(y_smoothed, score_min, score_max, y_degree)
    else:
        yfreq = _freq_from_raw(y, score_min, score_max)
        
    # ------------------------------------------------------------------
    # Calculate SEE
    # ------------------------------------------------------------------
    #Sample sizes
    nX = xfreq.sum()
    nY = yfreq.sum()
    
    #Probabilities
    f_x = (xfreq / nX).to_numpy()
    g_y = (yfreq / nY).to_numpy()
    
    #Cumulative distributions
    Fx = np.cumsum(f_x)
    Gy = np.cumsum(g_y)
    
    #Percentiles
    Px = np.zeros(len(f_x))
    
    for i in range(len(f_x)):
        if i == 0:
            Px[i] = f_x[i] / 2.0
        else:
            Px[i] = Fx[i-1] + f_x[i] / 2.0
    
    Gy_100 = Gy * 100
    
    se = np.zeros(len(f_x))
    for j in range(len(f_x)):
        k = min(int(np.searchsorted(Gy_100, Px[j]*100, side = "left")), len(Gy_100) - 1)
        Gy_star = Gy[k]
        
        #Cumulative probabilities
        Ghi = Gy[k]  
        Glo = Gy[k-1] if k > 0 else 0.0 
        
        #Density interval: G(y*) - G(y*-1)
        g0 = max((Ghi - Glo), 1e-10) #Prevent division by zero
        
        #Separate the terms to make math easier
        term1 = (1 - Px[j]) * Px[j] / (nX * g0**2)
        term2 = (1 / (nY * g0**2)) * (Glo - Px[j]**2 + ((Px[j] - Glo)**2) / g0)
        var_eY = term1 + term2
        se[j] = np.sqrt(max(var_eY, 0.0)) #Prevent zero again
        
    equated_df['SEE'] = se
    
    return equated_df
        

            
#SE = sqrt((1-q)*q/(n_x*g0^2) + (1/(n_y*g0^2))*(gm - q^2 + ((q-gm)^2)/g0)) - this is how equate calculated it; used a density interval
#gm = Glo
#TODO: incorporate into equating functions

    