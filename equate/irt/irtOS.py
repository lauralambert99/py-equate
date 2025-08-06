# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:44:06 2025

@author: laycocla
"""
import numpy as np
import pandas as pd

from .irt_helper import lord_wingersky_distribution, gauss_hermite_normal, cdf_mapping


def irtOS(formX_params, formY_params, theta_points=31, w1=0.5, model='2pl', form='parameters'):
    """
    Perform IRT Observed Score Equating.

    Parameters:
    - formX_params: DataFrame with item parameters in columns ('item_id', 'a', 'b', 'c') for Form X
    - formY_params: DataFrame with item parameters in columns ('item_id', 'a', 'b', 'c') for Form Y
    - theta_points: Number of points for quadrature integration (default 31)
    - w1: Weight for Population 1 (default 0.5)
    - model: '1pl', '2pl', or '3pl'
    - form: Must be 'parameters' for now

    Returns:
    - DataFrame with columns: 'X', 'eyx' (equated score)
    """
    if form != 'parameters':
        raise ValueError("Currently only 'parameters' form is supported.")

    paramsX = formX_params.copy()
    paramsY = formY_params.copy()

    #Check for c (set to 0 if missing)
    for df in [paramsX, paramsY]:
        if 'c' not in df.columns:
            df['c'] = 0.0

    #Generate Gauss-Hermite nodes and weights
    theta, weights = gauss_hermite_normal(theta_points)

    #Calculate population-weighted score distributions
    w2 = 1 - w1

    f1_X = w1 * lord_wingersky_distribution(paramsX, theta, weights, model) + \
           w2 * lord_wingersky_distribution(paramsX, theta, weights, model)

    f2_Y = w1 * lord_wingersky_distribution(paramsY, theta, weights, model) + \
           w2 * lord_wingersky_distribution(paramsY, theta, weights, model)
  

    #Perform equipercentile equating
    cdf_map = cdf_mapping(f1_X, f2_Y)

    return cdf_map



