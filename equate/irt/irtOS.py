# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:44:06 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import fftconvolve
from py_irt.scoring import calculate_theta

def irt_prob(theta, a, b, c, model='2pl'):
    """Return item probabilities at one theta point"""
    z = a * (theta - b)
    if model == '1pl':
        return 1 / (1 + np.exp(-z))
    elif model == '2pl':
        return 1 / (1 + np.exp(-z))
    elif model == '3pl':
        return c + (1 - c) / (1 + np.exp(-z))

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

def irtOS(formX_params, formY_params, w1 = 0.5, model = '2pl', form = 'parameters'):
    """
    Perform IRT Observed Score Equating.

    Parameters:
    - formX_params: DataFrame with item parameters ('item_id', 'a', 'b', 'c') for Form X
    - formY_params: DataFrame with item parameters ('item_id', 'a', 'b', 'c') for Form Y
    - w1: Weight for Population 1 (default 0.5)
    - model: '1pl', '2pl', or '3pl'
    - form: Must be 'parameters' for now (future: incorporate raw scores)

    Returns:
    - DataFrame with columns: 'X', 'eyx' (equated score)
    """

    if form != 'parameters':
        raise ValueError("Currently only 'parameters' form is supported. Please provide IRT parameters from py-irt, mirt, or another software.")

    paramsX = formX_params.copy()
    paramsY = formY_params.copy()

    #Check for c (set to 0 if missing)
    for df in [paramsX, paramsY]:
        if 'c' not in df.columns:
            df['c'] = 0.0

    #Make the theta grid and population distributions
    theta = np.linspace(-4, 4, 61)  #Future: make this customizable
    pdf = norm.pdf
    
    phi1 = pdf(theta)
    phi2 = pdf(theta)
    
    phi1 /= phi1.sum()
    phi2 /= phi2.sum()
    
    w2 = 1 - w1

    f1_X = w1 * score_distribution(paramsX, theta, phi1, model) + w2 * score_distribution(paramsX, theta, phi2, model)
    f2_Y = w1 * score_distribution(paramsY, theta, phi1, model) + w2 * score_distribution(paramsY, theta, phi2, model)

    def equate_scores(pmf_X, pmf_Y):
        f1_X_cum = pmf_X.cumsum()
        f2_Y_cum = pmf_Y.cumsum()
        eq_table = []

        for score_x, p in f1_X_cum.items():
            score_y = np.searchsorted(f2_Y_cum.values, p)
            eq_table.append((score_x, score_y))

        return pd.DataFrame(eq_table, columns=["X", "eyx"])

    return equate_scores(f1_X, f2_Y)



