# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 05:32:50 2025

@author: laycocla
"""
import numpy as np
import pandas as pd

def eq_see_asy(freq_X, freq_Y):
    """
    Calculate Standard Errors of Equating (SEE) for equipercentile equating
    using asymptotic SEE formula.
    
    Parameters
    ----------
    freq_X : pd.Series
        Frequencies for scores on form X, indexed by score (ascending).
    freq_Y : pd.Series
        Frequencies for scores on form Y, indexed by score (ascending).
        
    Returns
    -------
    pd.DataFrame with columns:
        - score_X : score on form X
        - equated_Y : equated score on form Y
        - SEE : standard error of equating
    """

    scores = freq_X.index.to_numpy()
    
    #Probabilities
    f_x = freq_X / freq_X.sum()
    g_y = freq_Y / freq_Y.sum()
    
    #Cumulative distributions
    Fx = f_x.cumsum()
    Gy = g_y.cumsum()
    Px = 100 * (Fx.shift(1, fill_value=0) + f_x / 2)
    Gy_100 = (Gy * 100).to_numpy()
    
    #Sample sizes
    nX, nY = freq_X.sum(), freq_Y.sum()
    
    def find_Y_star(px_val, Gy_val):
        idx = np.searchsorted(Gy_val, px_val, side="left")
        return min(idx, len(Gy_val)-1)

    e_yx, sees = [], []
    
    #This is the equating bit
    for i, px in enumerate(Px.to_numpy()):
        k = find_Y_star(px, Gy_100)
        Ghi = Gy.iloc[k]
        Glo = Gy.iloc[k-1] if k > 0 else 0.0
        
        if (Ghi <= Glo) or np.isclose(Ghi, Glo):
            yhat = scores[k].astype(float)
        else:
            yhat = (scores[k] - 0.5) + ((px/100.0 - Glo) / (Ghi - Glo))
            
        e_yx.append(float(yhat))
        
        #This is the SEE bit
        Fx_val = Px.iloc[i] / 100
        Fy_val = Gy.iloc[k]
        fY_val = g_y.iloc[k]
        
        if fY_val > 0:
            var = ((Fx_val*(1 - Fx_val))/(nX * (fY_val**2))) + \
                  ((Fy_val*(1 - Fy_val))/(nY * (fY_val**2)))
            see = np.sqrt(var)
        else:
            see = np.nan #Can't divide by zero
        sees.append(see)

    equated_df = pd.DataFrame({
        'score': scores,
        'equated': e_yx,
        'SEE': sees
    })
    
    return equated_df
    
    