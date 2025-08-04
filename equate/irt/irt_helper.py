# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:30:48 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

#Used to make theta grid
def irt_prob(theta, a, b, c, model='2pl'):
    """Return item probabilities at one theta point"""
    z = a * (theta - b)
    if model == '1pl':
        return 1 / (1 + np.exp(-z))
    elif model == '2pl':
        return 1 / (1 + np.exp(-z))
    elif model == '3pl':
        return c + (1 - c) / (1 + np.exp(-z))

#Lord - Wingersky stuff
def score_distribution(params, theta_grid, weights, model='2pl'):
    score_max = params.shape[0]
    score_pmf = np.zeros(score_max + 1)

    for i, theta in enumerate(theta_grid):
        p_i = irt_prob(theta, params['a'].values, params['b'].values, params['c'].values, model)
        dist_theta = np.array([1.0])
        for p in p_i:
            dist_theta = fftconvolve(dist_theta, [1 - p, p])[:len(dist_theta)+1] # recursion
        score_pmf[:len(dist_theta)] += weights[i] * dist_theta

    return pd.Series(score_pmf, index=np.arange(len(score_pmf)))

