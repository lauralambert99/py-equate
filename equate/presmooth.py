# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:19:35 2025

@author: laycocla
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import chi2



def _fit_glm(design: pd.DataFrame, counts: np.ndarray) -> sm.GLMResultsWrapper:
    """
    Fit a Poisson log-linear model using IRLS.

    `design` should NOT include an intercept column

    """
    X = sm.add_constant(design, has_constant = "add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(
            counts,
            X,
            family = sm.families.Poisson(),
        )
        result = model.fit(maxiter = 100, tol = 1e-8, disp = False)
    return result


def presmooth(freq, score_min, score_max, max_order=10):
    """
    Perform log-linear Poisson presmoothing for a single form using polynomial
    orders 1 to max_order.

    Parameters
    ----------
    freq : array-like
        Observed frequencies for a test, indexed by score.
    score_min : int
        Minimum possible score.
    score_max : int
        Maximum possible score.
    max_order : int, optional
        Maximum polynomial degree to try (default 10).

    Returns
    -------
    presmooth_df : pd.DataFrame
        Columns: ['score', 'observed', 'Order_1', 'Order_2', ..., 'Order_N']
        Each Order_N column contains the smoothed frequencies for that
        polynomial order.
    """
    score_range = range(score_min, score_max + 1)
    freq = pd.Series(freq).reindex(score_range, fill_value=0)
    scores = np.arange(score_min, score_max + 1, dtype=float)
    counts = freq.values.astype(float)
    
    #Scale scores first - avoid crazy matrices
    score_mean = scores.mean()
    score_sd   = scores.std(ddof=1)
    scores_scaled = (scores - score_mean) / score_sd

    result_df = pd.DataFrame({
        "score": scores.astype(int),
        "observed": counts,
    })

    for order in range(1, max_order + 1):
        #Design matrix: intercept + scores^1 + ... + scores^order
        poly_terms = np.vstack([scores_scaled**i for i in range(1, order + 1)]).T
        intercept = np.ones((len(scores), 1))
        X_design = np.hstack([intercept, poly_terms])

        params, _ = _fit_poisson_loglinear(X_design, counts)
        result_df[f"Order_{order}"] = np.exp(X_design @ params)

    return result_df