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
        probs[1:] = probs[1:] * (1 - p) + probs[:-1] * p
        probs[0] *= (1 - p)

    return probs

def lord_wingersky_distribution(params, theta_grid, model='3pl', D=1.7):
    """
    Returns a 2D array: rows = raw scores (0..n_items), columns = theta points.
    """
    if 'c' not in params.columns:
        params['c'] = 0.0

    a, b, c = params['a'].values, params['b'].values, params['c'].values
    n_items = len(a)

    def get_fn(a, b, c):
        return lambda t: c + (1 - c) / (1 + np.exp(-D * a * (t - b)))

    irt_fns = [get_fn(a[i], b[i], c[i]) for i in range(n_items)]

    score_matrix = np.array([lord_wingersky(t, irt_fns) for t in theta_grid])  # shape = (n_theta, n_scores)
    return score_matrix.T  # shape = (n_scores, n_theta)


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
