# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:44:06 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from .irt_helper import lord_wingersky_distribution, gauss_hermite_quadrature
from .. import equipercen 

def irtOS(formX_params, formY_params, theta_points=31, w1=0.5, model='2pl', form='parameters', n_sample=10000):
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
        IRT model ('1pl', '2pl', '3pl').
    form : str
        Must be 'parameters'.
    n_sample : int
        Number of synthetic examinees to generate for equipercentile equating.
    
    Returns
    -------
    DataFrame with columns: 'Theta', 'Scale', 'Equated'
    """
    if form != 'parameters':
        raise ValueError("Only 'parameters' form is currently supported.")

    #Ensure c column exists
    for df in [formX_params, formY_params]:
        if 'c' not in df.columns:
            df['c'] = 0.0

    #Generate theta grid and weights
    theta, weights = gauss_hermite_quadrature(theta_points)
    w2 = 1 - w1

    #Compute score distributions: rows = scores, cols = theta
    px_theta = lord_wingersky_distribution(formX_params, theta, model=model)
    py_theta = lord_wingersky_distribution(formY_params, theta, model=model)

    #Compute marginal PMFs across raw scores
    f1_X = w1 * np.dot(px_theta, weights) + w2 * np.dot(px_theta, weights)
    f2_Y = w1 * np.dot(py_theta, weights) + w2 * np.dot(py_theta, weights)

    scores = np.arange(len(f1_X))

    #Compute expected theta per raw score
    theta_x = []
    for i, score in enumerate(scores):
        weighted_prob = px_theta[i, :] * weights
        if weighted_prob.sum() == 0:
            theta_x.append(theta[0] if i == 0 else theta[-1])
        else:
            theta_x.append(np.sum(theta * weighted_prob) / weighted_prob.sum())

    #Convert PMFs to synthetic raw score vectors
    #Here's where some of the slight discrepancies may still remain
    def sample_scores_from_pmf(pmf, n_sample=n_sample):
        return np.random.choice(np.arange(len(pmf)), size=n_sample, p=pmf)

    x_vec = sample_scores_from_pmf(f1_X, n_sample)
    y_vec = sample_scores_from_pmf(f2_Y, n_sample)
    
    x_vec1 = pd.Series(x_vec)
    y_vec1 = pd.Series(y_vec)

    #Perform equipercentile equating
    eq_result = equipercen(x_vec1, y_vec1, score_min=0, score_max=len(f1_X)-1)

    #Construct output DataFrame
    out = pd.DataFrame({
        'Theta': theta_x,
        'Scale': scores,
        'Equated': eq_result['equated']
    })

    return out
