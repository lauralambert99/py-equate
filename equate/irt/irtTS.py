# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:00:41 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .irt_helper import gauss_hermite_quadrature, ts_curve 

def irtTS(formX_params, formY_params, score_range=None, model='2pl', theta_points=31):
    """
    Perform IRT True Score Equating, aligned with irtOS theta grid.
    
    Parameters:
    - formX_params: DataFrame with item parameters ('a', 'b', 'c') for Form X
    - formY_params: DataFrame with item parameters ('a', 'b', 'c') for Form Y
    - score_range: Iterable of observed score values on Form X (e.g., range(0, n_items+1))
                   If None, inferred from Form X item count
    - model: IRT model ('1pl', '2pl', or '3pl')
    - theta_points: Number of points for Gauss-Hermite quadrature (default 31)
    
    Returns:
    - DataFrame with columns: 'X' (Form X score), 'Theta' (associated theta), 'tyx' (equated true score)
    """
    # 1. Generate theta grid using Gauss-Hermite quadrature
    theta, weights = gauss_hermite_quadrature(theta_points)

    # 2. Compute expected true scores for each theta
    T_X = ts_curve(formX_params, theta, model=model)
    T_Y = ts_curve(formY_params, theta, model=model)

    # 3. Create interpolation functions
    theta_from_Tx = interp1d(T_X, theta, bounds_error=False, fill_value="extrapolate")
    Ty_from_theta = interp1d(theta, T_Y, bounds_error=False, fill_value="extrapolate")

    # 4. Determine score range
    if score_range is None:
        score_max = formX_params.shape[0]  # assumes one row per item
        score_range = np.arange(0, score_max + 1)

    # 5. Compute equated scores and associated thetas
    tyx = []
    thetas = []
    for x in score_range:
        theta_x = theta_from_Tx(x)
        y = Ty_from_theta(theta_x)
        thetas.append(theta_x)
        tyx.append(y)

    # 6. Build output DataFrame
    out = pd.DataFrame({
        "X": score_range,
        "Theta": thetas,
        "tyx": tyx
    })

    return out


#Future TODO:  Add theta_range, num_grid_points to fxn args
