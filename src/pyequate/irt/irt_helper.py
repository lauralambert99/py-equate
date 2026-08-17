# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:30:48 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

def lord_wingersky(theta, irt_functions):
    """
    Compute the distribution of raw scores for a single theta using Lord-Wingersky recursion.
    """
    n_items = len(irt_functions)
    probs = np.zeros(n_items + 1)
    probs[0] = 1.0

    for j in range(n_items):
        p = irt_functions[j](theta)
        new_probs = probs.copy()
        for r in range(1, j + 2):
            new_probs[r] = probs[r] * (1 - p) + probs[r-1] * p
        new_probs[0] = probs[0] * (1 - p)
        probs = new_probs
    return probs

def lord_wingersky_distribution(params, theta_grid, model='2pl', D=1.7):
    """
    Returns a 2D array: rows = raw scores (0..n_items), columns = theta points.
    """
    if 'c' not in params.columns:
        params['c'] = 0.0

    a, b, c = params['a'].values, params['b'].values, params['c'].values
    n_items = len(a)

    def get_fn(a_i, b_i, c_i):
        if model == '1pl':
            return lambda t: c_i + (1 - c_i) / (1 + np.exp(-D * (t - b_i)))
        elif model == '2pl':
            return lambda t: c_i + (1 - c_i) / (1 + np.exp(-D * a_i * (t - b_i)))
        elif model == '3pl':
            return lambda t: c_i + (1 - c_i) / (1 + np.exp(-D * a_i * (t - b_i)))
        else:
            raise ValueError(f"Unknown model: {model}")

    irt_fns = [get_fn(a[i], b[i], c[i]) for i in range(n_items)]

    score_matrix = np.array([lord_wingersky(t, irt_fns) for t in theta_grid])
    
    return score_matrix.T 

def gauss_quad_prob(n, mu=0.0, sigma=1.0):
    """
    Generate Gaussian quadrature nodes and weights
    
    """
    from scipy.special import roots_hermitenorm
    
    #Get roots and weights for probabilist's Hermite polynomials
    x, w = roots_hermitenorm(n)
    
    #The nodes are already correct for standard normal integration
    nodes = x
    
    #Weights need to be normalized for probability (integral = 1)
    #The weights from roots_hermitenorm integrate exp(-x^2/2) over (-inf, inf)
    #which equals sqrt(2*pi), so divide by that
    weights = w / np.sqrt(2 * np.pi)
    
    #Scale to N(mu, sigma^2)
    nodes = mu + sigma * nodes
    
    return nodes, weights

def gauss_hermite_quadrature(n_points):
    """
    Generate Gauss-Hermite quadrature points and weights for normal distribution integration.

    Returns
    -------
    theta : ndarray
        Quadrature points.
    weights : ndarray
        Corresponding weights, normalized to sum to 1.
    """
    #Hermite nodes and weights
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    
    #Scale nodes for standard normal
    theta = nodes * np.sqrt(2)  
    
    #Normalize weights
    weights = weights / np.sqrt(np.pi)
    weights /= weights.sum()
    
    return theta, weights

def cdf_mapping(fx, fy, scores_x=None, scores_y=None):
    """Equipercentile mapping with linear interpolation for decimal scores."""
    if scores_x is None:
        scores_x = np.arange(len(fx))
    if scores_y is None:
        scores_y = np.arange(len(fy))

    Fx = np.cumsum(fx)
    Gy = np.cumsum(fy)

    eyx = np.interp(Fx, Gy, scores_y)  # linear interpolation
    return pd.DataFrame({'X': scores_x, 'eyx': eyx})

def irt_prob(theta, a, b, c=None, model='2pl', D=1.7):
    """
    Compute probability of a correct response for given theta and item parameters.

    Parameters:
    - theta: scalar or array of ability values
    - a: discrimination parameter(s)
    - b: difficulty parameter(s)
    - c: guessing parameter(s), optional (default 0)
    - model: '1pl', '2pl', or '3pl'

    Returns:
    - Probability (same shape as theta)
    """
    if c is None:
        c = np.zeros_like(a)
    
    if model == '1pl':
        P = 1 / (1 + np.exp(-D * (theta - b)))  # a=1
    elif model == '2pl':
        P = 1 / (1 + np.exp(-D * a * (theta - b)))
    elif model == '3pl':
        P = c + (1 - c) / (1 + np.exp(-D * a * (theta - b)))
    else:
        raise ValueError(f"Unknown model: {model}")
    return P


def ts_curve(params, theta_grid, model='2pl', D=1.7):
    """
    Compute expected true scores for a given item parameter set across a theta grid.
    
    Parameters:
    - params: DataFrame with item parameters ('a', 'b', 'c')
    - theta_grid: array of theta values
    - model: '1pl', '2pl', '3pl'
    - D: scaling constant for logistic function (default 1.7)
    
    Returns:
    - Array of expected true scores for each theta
    """
    #Ensure c exists
    if 'c' not in params.columns:
        params['c'] = 0.0

    a = params['a'].values
    b = params['b'].values
    c = params['c'].values
    n_items = len(a)

    #Compute probability matrix: rows=theta, columns=items
    prob_matrix = np.array([irt_prob(theta, a, b, c, model=model, D=D) for theta in theta_grid])

    #True score = sum of expected correct probabilities per theta
    T_theta = prob_matrix.sum(axis=1)
    return T_theta

def equipercentile_irt(fx, fy):
    """
    Equipercentile equating for IRT observed scores.
    
    Parameters
    ----------
    fx : array-like
        PMF for Form X
    fy : array-like
        PMF for Form Y
    
    Returns
    -------
    array : Equated scores e_Y(x)
    """
    fx = np.array(fx)
    fy = np.array(fy)
    
    #Normalize
    fx = fx / fx.sum()
    fy = fy / fy.sum()
    
    n_x = len(fx)
    n_y = len(fy)
    
    #Compute CDFs
    Fx = np.cumsum(fx)
    Gy = np.cumsum(fy)
    
    Px = np.zeros(n_x)
    Px[0] = fx[0] / 2.0
    for i in range(1, n_x):
        Px[i] = Fx[i-1] + fx[i] / 2.0
    
    #Equipercentile equating: find y such that G(y) = F(x)
    eyx = np.zeros(n_x)
    
    for i in range(n_x):
        px = Px[i]
        
        # Find where px falls in Gy
        if px <= 0:
            eyx[i] = 0.0
        elif px >= 1.0:
            eyx[i] = float(n_y - 1)
        else:
            idx = np.searchsorted(Gy, px, side='left')
            
            if idx == 0:
                #px is below first cumulative probability
                eyx[i] = px / Gy[0] if Gy[0] > 0 else 0.0
            elif idx >= n_y:
                #px is above last cumulative probability
                eyx[i] = float(n_y - 1)
            else:
                Glo = Gy[idx - 1] if idx > 0 else 0.0
                Ghi = Gy[idx]
                
                if np.isclose(Ghi, Glo):
                    eyx[i] = float(idx)
                else:
                    eyx[i] = (idx - 0.5) + (px - Glo) / (Ghi - Glo)
    
    return eyx