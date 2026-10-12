# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:36:20 2025

@author: laycocla
"""
import numpy as np
import pandas as pd

def linear_se(formX, formY, x_scores = None):
    """
    Compute standard error of equating (SEE) for linear equating in a random groups design
    using the Delta method.

    Parameters:
    - formX: array-like raw scores from Form X
    - formY: array-like raw scores from Form Y
    - x_scores: array-like scores on Form X to compute SEE for (default: all unique scores in formX)

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

    if x_scores is None:
        x_scores = np.unique(formX)
    else:
        x_scores = np.asarray(x_scores)

    SEE = np.zeros_like(x_scores, dtype=float)

    for i, x in enumerate(x_scores):
        d_mu_y = 1
        d_mu_x = -sd_y / sd_x
        d_sd_y = (x - mu_x) / sd_x
        d_sd_x = -sd_y * (x - mu_x) / (sd_x ** 2)

        SEE[i] = np.sqrt(
            d_mu_x**2 * var_mu_x +
            d_mu_y**2 * var_mu_y +
            d_sd_x**2 * var_sd_x +
            d_sd_y**2 * var_sd_y
        )

    return pd.DataFrame({'scores': x_scores, 'SEE': SEE})