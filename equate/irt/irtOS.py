# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:44:06 2025

@author: laycocla
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import fftconvolve
from .irt_helper import irt_prob
from .irt_helper import score_distribution
from equate.equipercen import equipercen



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
    
    #Need score_min and score_max
    score_min = 0
    score_max = paramsX['item_id'].nunique()

    #Have PMFs, need raw scores
    #Can multiply by a big number to get?

    def pmf_to_scores(pmf, scale = 10000):
        counts = (pmf.values * scale).round().astype(int)
        scores = np.repeat(pmf.index.values, counts)
        return scores

    scores_x = pmf_to_scores(f1_X)
    scores_y = pmf_to_scores(f2_Y)
    
    #Call equipercen function from before
    eq_result = equipercen(scores_x, scores_y, score_min, score_max)

    return eq_result



