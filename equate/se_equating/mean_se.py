# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:07:56 2025

@author: Laura
"""
import numpy as np
import pandas as pd

def mean_equate_se(formX, formY):
    """
    Compute the standard error of equating (SEE) for mean equating
    in a random groups design using raw scores and the Delta Method

    Parameters:
    - formX: array-like raw scores from Form X
    - formY: array-like raw scores from Form Y

    Returns:
    - standard error of equating (float)
    """
    formX = np.asarray(formX)
    formY = np.asarray(formY)

    var_x = np.var(formX, ddof=1)
    var_y = np.var(formY, ddof=1)

    n_x = len(formX)
    n_y = len(formY)

    se2 = var_x / n_x + var_y / n_y
    return np.sqrt(se2)

def linear_equate_se(formX, formY, y_scores = None):
    """
    Compute standard error of equating (SEE) for linear equating in a random groups design
    using the Delta method.

    Parameters:
    - formX: array-like raw scores from Form X
    - formY: array-like raw scores from Form Y
    - y_scores: array-like scores on Form Y to compute SEE for (default: all unique scores in formY)

    Returns:
    - pd.Dataframe:
        'scores': y_scores,
        'SEE': standard errors of equating at each y_score
    """
    formX = np.asarray(formX)
    formY = np.asarray(formY)

    mu_x = np.mean(formX)
    mu_y = np.mean(formY)
    
    sd_x = np.std(formX, ddof=1)
    sd_y = np.std(formY, ddof=1)

    n_x = len(formX)
    n_y = len(formY)

    var_mu_x = sd_x**2 / n_x
    var_mu_y = sd_y**2 / n_y
    
    var_sd_x = (sd_x**2) / (2 * (n_x - 1))
    var_sd_y = (sd_y**2) / (2 * (n_y - 1))

    if y_scores is None:
        y_scores = np.unique(formY)
    else:
        y_scores = np.asarray(y_scores)

    SEE = np.zeros_like(y_scores, dtype=float)

    for i, y in enumerate(y_scores):
        d_mu_x = 1
        d_mu_y = -sd_x / sd_y
        d_sd_x = (y - mu_y) / sd_y
        d_sd_y = -sd_x * (y - mu_y) / (sd_y ** 2)

        SEE[i] = np.sqrt(
            d_mu_x**2 * var_mu_x +
            d_mu_y**2 * var_mu_y +
            d_sd_x**2 * var_sd_x +
            d_sd_y**2 * var_sd_y
        )

    return pd.DataFrame({'scores': y_scores, 'SEE': SEE})