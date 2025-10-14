# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:44:06 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from .irt_helper import lord_wingersky_distribution, gauss_hermite_quadrature
from .. import equipercen 

def irtOS(formX_params, formY_params, theta_points=10, w1=0.5, model='2pl', 
          mu=0.0, sigma=1.0, D=1.7, A=1.0, B=0.0):
    """
    Perform IRT Observed Score Equating using equipercentile equating.
    
    Parameters
    ----------
    formX_params : DataFrame
        Item parameters for Form X ('item_id', 'a', 'b', 'c').
    formY_params : DataFrame
        Item parameters for Form Y ('item_id', 'a', 'b', 'c').
    theta_points : int
        Number of quadrature points.
    w1 : float
        Weight for population 1.
    model : str
        IRT model ('1pl', '2pl', '3pl')
    mu : float
        Mean of ability distribution (default = 0.0)
    sigma : float
        SD of ability distribution (default = 1.0)
    D : float
        Scaling constant (default = 1.7)
    A : float
        Scale linking parameter (default = 1.0)
    B : float
        Scale linking parameter (default = 0.0)
    
    Returns
    -------
    DataFrame with columns: 
        - 'Scale' : Raw scores on Form X
        - 'eyx' : Equated scores
        - 'f_hat' : Form X PMF
        - 'g_hat' : Form Y PMF
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
            
    #Ensure c column exists
    for df in [formX_params, formY_params]:
        if 'c' not in df.columns:
            df['c'] = 0.0

    #Generate theta quadrature points and weights
    theta, weights = gauss_quad_prob(n_points, mu=mu, sigma=sigma)
    w2 = 1 - w1
    
    #Compute score distributions using Lord-Wingersky
    #Want rows = scores, cols = theta
    px_theta = lord_wingersky_distribution(formX_params, theta, model=model, D=D)
    py_theta = lord_wingersky_distribution(formY_params, theta, model=model, D=D)

    #Use params to get number of items on each form
    if isinstance(formX_params, dict):
        n_items_x = len(formX_params['a'])
        n_items_y = len(formY_params['a'])
    else:
        n_items_x = len(formX_params)
        n_items_y = len(formY_params)

    #Integrate over theta to get marginal PMFs
    #NEW
    f_hat = np.dot(px_theta, weights)
    g_hat = np.dot(py_theta, weights)
    
    #Normalize just in case
    f_hat = f_hat / f_hat.sum()
    g_hat = g_hat / g_hat.sum()

    #Now, equipercentile equating
    eYx = equipercen(f_hat, g_hat)
        
    # Return results in R-compatible format
    result = {
        'Scale': np.arange(len(f_hat)),
        'eYx': eYx,
        'f_hat': f_hat,
        'g_hat': g_hat    
    }
        
    return result

