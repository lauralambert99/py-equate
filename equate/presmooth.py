# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:19:35 2025

@author: laycocla
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:19:35 2025
@author: laycocla

Revised: statsmodels replaced with scipy for Poisson log-linear presmoothing.
         Single-form input; output is wide-format with one column per order.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _fit_poisson_loglinear(X_design, counts):
    """
    Fit a Poisson log-linear model via maximum likelihood using scipy.

    Minimizes the negative log-likelihood:
        -sum(y * (X @ beta) - exp(X @ beta))

    Parameters
    ----------
    X_design : np.ndarray, shape (n, p)
        Design matrix (intercept + polynomial terms).
    counts : np.ndarray, shape (n,)
        Observed frequency counts.

    Returns
    -------
    params : np.ndarray
        Fitted coefficients.
    success : bool
    """
    n_params = X_design.shape[1]

    def neg_loglik(beta):
        eta = np.clip(X_design @ beta, -500, 500)
        return -np.sum(counts * eta - np.exp(eta))

    def grad(beta):
        eta = np.clip(X_design @ beta, -500, 500)
        return -X_design.T @ (counts - np.exp(eta))

    result = minimize(
        neg_loglik,
        x0=np.zeros(n_params),
        jac=grad,
        method="L-BFGS-B",
    )
    if result.success:
        return result.x, True
    # Retry with tighter tolerance on failure
    result2 = minimize(
        neg_loglik,
        x0=result.x,
        jac=grad,
        method="L-BFGS-B",
        options={"ftol": 1e-9, "gtol": 1e-6, "maxiter": 500},
    )
    return result2.x, result2.success


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