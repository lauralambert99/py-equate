# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:00:41 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .irt_helper import ts_curve 

def irtTS(formX_params, formY_params, score_range=None, model='2pl', theta_points=30, theta_min=-4, theta_max=4, D=1.7, A=1.0, B=0.0):
    """
    Perform IRT True Score Equating.
    
    Parameters:
    - formX_params: DataFrame with item parameters ('a', 'b', 'c') for Form X
    - formY_params: DataFrame with item parameters ('a', 'b', 'c') for Form Y
    - score_range: Iterable of observed score values on Form X (e.g., range(0, n_items+1))
                   If None, inferred from Form X item count
    - model: IRT model ('1pl', '2pl', or '3pl')
    - theta_points: Number of points for theta grid (default 30)
    - theta_min: Lower bound of theta in theta grid (default -4)
    - theta_max: Upper bound of theta in theta grid (default 4)
    - A : float
        Scale linking parameter (default = 1.0)
    - B : float
        Scale linking parameter (default = 0.0)
    
    Returns:
    - DataFrame with columns: 'X' (Form X score), 'Theta' (associated theta), 'tyx' (equated true score)
    """
    #First, transformation
    if A != 1.0 or B != 0.0:
        if isinstance(formY_params, dict):
            formY_params = formY_params.copy()
            formY_params['b'] = [A * b + B for b in formY_params['b']]
            formY_params['a'] = [a / A for a in formY_params['a']]
        else:
            formY_params = formY_params.copy()
            formY_params['b'] = A * formY_params['b'] + B
            formY_params['a'] = formY_params['a'] / A
    
    #Generate theta grid using Gauss-Hermite quadrature
    theta = np.linspace(theta_min, theta_max, theta_points)

    #Compute expected true scores for each theta
    T_X = ts_curve(formX_params, theta, model=model, D=D)
    T_Y = ts_curve(formY_params, theta, model=model, D=D)

    #Create interpolation functions
    theta_from_Tx = interp1d(T_X, theta, kind='linear',
                             bounds_error=False, fill_value="extrapolate")
    Ty_from_theta = interp1d(theta, T_Y, kind='linear',
                             bounds_error=False, fill_value="extrapolate")

    #Determine score range
    if score_range is None:
        score_max = formX_params.shape[0]  # assumes one row per item
        score_range = np.arange(0, score_max + 1)

    #Compute equated scores and associated thetas
    tyx = []
    thetas = []
    for x in score_range:
        theta_x = theta_from_Tx(x)
        y = Ty_from_theta(theta_x)
        thetas.append(theta_x)
        tyx.append(y)

    #Build output DataFrame
    out = pd.DataFrame({
        "X": score_range,
        "Theta": thetas,
        "tyx": tyx
    })

    return out
