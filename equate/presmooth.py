# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:19:35 2025

@author: laycocla
"""

import numpy as np
import pandas as pd
import statsmodels.tools as smt
import statsmodels.api as sm
import warnings

from typing import Literal
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from scipy.optimize import minimize
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Single Poisson GLM fit function
# ---------------------------------------------------------------------------
def _fit_glm(design: pd.DataFrame, counts: np.ndarray) -> GLMResultsWrapper:
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

# ---------------------------------------------------------------------------
# Orthogonal polynomial basis matrix
# ---------------------------------------------------------------------------
#Need this to prevent multicollinearity in higher-order terms
#Mimicing poly() from R

def _ortho_poly(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute an orthogonal polynomial basis for `x` up to `degree`, using
    the Gram-Schmidt process.

    Parameters
    ----------
    x : 1-D array of predictor values (e.g. integer scores).
    degree : maximum polynomial degree.

    Returns
    -------
    Z : ndarray, shape (len(x), degree + 1)
        Column 0 is the intercept (all ones); columns 1..degree are
        orthogonal polynomial contrasts.
    """
    n = len(x)
    Z = np.zeros((n, degree + 1))
    Z[:, 0] = 1.0  #Intercept

    if degree == 0:
        return Z

    #Gram-Schmidt orthogonalisation bit
    Z[:, 1] = x - x.mean()

    for k in range(2, degree + 1):
        z_prev  = Z[:, k - 1]
        z_prev2 = Z[:, k - 2]

        v = x * z_prev
        alpha = np.dot(v, z_prev) / np.dot(z_prev, z_prev)
        beta  = np.dot(v, z_prev2) / np.dot(z_prev2, z_prev2)

        Z[:, k] = v - alpha * z_prev - beta * z_prev2

    #Normalise columns 1+ to unit norm
    for k in range(1, degree + 1):
        nrm = np.linalg.norm(Z[:, k])
        if nrm > 0:
            Z[:, k] /= nrm

    return Z


# ---------------------------------------------------------------------------
# Overall function
# ---------------------------------------------------------------------------
def presmooth(
        freq: "array-like", 
        score_min: int, 
        score_max: int, 
        degrees: list[int] | None = None,
        scorefun: pd.DataFrame | None = None,
        compare: bool = False,
        choose: bool = False,
        choosemethod: Litera["chi", "aic", "bic"] = "aic",
        chip: float = 0.05,
        verbose: bool = False) -> pd.DataFrame | pd.Series | dict:
    """
    Perform log-linear Poisson presmoothing for a single form using polynomial
    orders.

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