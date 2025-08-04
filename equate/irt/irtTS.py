# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:00:41 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .irt_helper import irt_prob

#Future TODO:  Add theta_range, num_grid_points to fxn args

#Compute true scores over theta grid
def ts_curve(params, theta_grid, model='2pl'):
    prob_matrix = np.array([
        irt_prob(theta, params['a'].values, params['b'].values, params['c'].values, model)
        for theta in theta_grid
    ])
    return prob_matrix.sum(axis = 1)  # sum of probabilities across items

def irtTS(paramsX, paramsY, score_range=None, model='2pl', theta_grid=None):
    """
    Perform IRT True Score Equating.
    
    Parameters:
    - paramsX: DataFrame with item parameters ('a', 'b', 'c') for Form X
    - paramsY: DataFrame with item parameters ('a', 'b', 'c') for Form Y
    - score_range: Optional iterable of observed score values on Form X (e.g., range(0, 41))
                   If None, it's inferred from Form X item count
    - model: IRT model ('1pl', '2pl', or '3pl')
    - theta_grid: Optional custom grid (default: np.linspace(-4, 4, 501))

    Returns:
    - DataFrame with columns: 'X', 'tyx' (true score equated from Form X to Y)
    """
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 501)
        
    T_X = ts_curve(paramsX, theta_grid, model)
    T_Y = ts_curve(paramsY, theta_grid, model)

    #Interpolation functions
    theta_from_Tx = interp1d(T_X, theta_grid, bounds_error = False, fill_value = "extrapolate")
    Ty_from_theta = interp1d(theta_grid, T_Y, bounds_error = False, fill_value = "extrapolate")

    #Score range
    if score_range is None:
        score_max = paramsX.shape[0]  # assumes one row per item
        score_range = np.arange(0, score_max + 1)

    #Do equating
    tyx = []
    for x in score_range:
        theta = theta_from_Tx(x)
        y = Ty_from_theta(theta)
        tyx.append((x, y))

    return pd.DataFrame(tyx, columns=["X", "tyx"])
