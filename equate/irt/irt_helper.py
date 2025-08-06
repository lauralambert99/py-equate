# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:30:48 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

def lord_wingersky(theta, irt_functions):
    n_items = len(irt_functions)
    max_score = n_items
    probs = np.zeros((n_items + 1,))
    probs[0] = 1.0

    for j in range(n_items):
        p = irt_functions[j](theta)
        probs[1:] = probs[1:] * (1 - p) + probs[:-1] * p
        probs[0] *= (1 - p)

    return probs

def lord_wingersky_distribution(params, theta_grid, weights, model='3pl', D=1.7):
    def get_fn(a, b, c):
        return lambda t: c + (1 - c) / (1 + np.exp(-D * a * (t - b)))

    if 'c' not in params.columns:
        params['c'] = 0.0

    a, b, c = params['a'].values, params['b'].values, params['c'].values
    n_items = len(a)

    irt_fns = [get_fn(a[i], b[i], c[i]) for i in range(n_items)]
    score_matrix = np.array([lord_wingersky(t, irt_fns) for t in theta_grid])
    pmf = np.dot(weights, score_matrix)

    return pmf

def gauss_hermite_normal(n_points):
    nodes, weights = hermgauss(n_points)

    theta = nodes * np.sqrt(2)
    weights = weights / np.sqrt(np.pi)

    #Weights need to sum to 1 
    weights /= weights.sum()

    return theta, weights

def cdf_mapping(fx, fy):
    Fx = np.cumsum(fx)
    Gy = np.cumsum(fy)
    y_scores = np.arange(len(fy))

    def map_score(p):
        return y_scores[np.searchsorted(Gy, p, side='left')]

    X = np.arange(len(fx))
    eyx = [map_score(p) for p in Fx]

    return pd.DataFrame({'X': X, 'eyx': eyx})