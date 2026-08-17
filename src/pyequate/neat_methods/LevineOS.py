# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:05:02 2025

@author: laycocla
"""

import pandas as pd
import numpy as np

from ..helper_fxns.scores_from_items import scores_from_items

def LevineOS(x=None, y=None, common_x=None, common_y=None, scores=None, w1=0.5,
             anchor="internal", items_x=None, items_y=None, common=None):
    """
    Perform Levine Observed Score equating.

    Provide EITHER total scores OR item-level responses:

    Total-score input
    -----------------
    x, y : pandas.Series
        Raw total scores for Form X (population 1) and Form Y
        (population 2). With anchor="internal" these totals must
        INCLUDE the common items; with anchor="external" they must not.
    common_x, common_y : pandas.Series
        Anchor scores, paired row-wise with x and y respectively

    Item-level input
    ----------------
    items_x, items_y : pandas.DataFrame or 2D array-like
        Item response matrices (rows = examinees, columns = items) for
        the Form X and Form Y groups.
    common : array-like
        Column names or integer positions of the common items (must
        identify the same items in both matrices).

    Other parameters
    ----------------
    scores : array-like
        Score points on Form X to be equated. Required.
    w1 : float, optional
        Synthetic population weight for group 1. Defaults to 0.5.
    anchor : str, optional
        "internal" or "external". Defaults to "internal".

    Returns
    -------
    pandas.DataFrame
        Columns 'Scores' and 'ey' (Levine OS equivalents on the Y scale).
    """

    #Data checks
    valid_anchor = ["internal", "external"]
    if anchor not in valid_anchor:
        raise ValueError(f"Anchor '{anchor}' not supported. Choose from {valid_anchor}")

    if scores is None:
        raise ValueError("`scores` (the Form X score points to equate) is required.")

    item_input = items_x is not None or items_y is not None or common is not None
    total_input = x is not None or y is not None or common_x is not None or common_y is not None

    if item_input and total_input:
        raise ValueError("Provide either total-score input (x, y, common_x, common_y) "
                         "or item-level input (items_x, items_y, common), not both.")

    if item_input:
        if items_x is None or items_y is None or common is None:
            raise ValueError("Item-level input requires items_x, items_y, and common.")
        x, common_x = scores_from_items(items_x, common, anchor=anchor)
        y, common_y = scores_from_items(items_y, common, anchor=anchor)
    else:
        if x is None or y is None or common_x is None or common_y is None:
            raise ValueError("Total-score input requires x, y, common_x, and common_y.")
        x = pd.Series(x).astype(float)
        y = pd.Series(y).astype(float)
        common_x = pd.Series(common_x).astype(float)
        common_y = pd.Series(common_y).astype(float)

    if len(x) != len(common_x):
        raise ValueError("x and common_x must be paired (same length/examinees).")
    if len(y) != len(common_y):
        raise ValueError("y and common_y must be paired (same length/examinees).")

    #Weights 
    w2 = (1 - w1)

    #Gamma
    cov_x = x.cov(common_x)
    cov_y = y.cov(common_y)

    if cov_x == 0 or cov_y == 0:
        raise ValueError("Zero covariance between total and anchor scores; "
                         "gamma terms are undefined.")

    if anchor == "internal":
        gamma_1 = x.var(ddof=1) / cov_x
        gamma_2 = y.var(ddof=1) / cov_y
    else:  # external
        gamma_1 = (x.var(ddof=1) + cov_x) / (common_x.var(ddof=1) + cov_x)
        gamma_2 = (y.var(ddof=1) + cov_y) / (common_y.var(ddof=1) + cov_y)

    #Synthetic population bits
    mu_sx = np.mean(x) - w2 * gamma_1 * (np.mean(common_x) - np.mean(common_y))
    mu_sy = np.mean(y) + w1 * gamma_2 * (np.mean(common_x) - np.mean(common_y))

    var_sx = (x.var(ddof=1)
              - w2 * (gamma_1**2) * (common_x.var(ddof=1) - common_y.var(ddof=1))
              + w1 * w2 * (gamma_1**2) * (np.mean(common_x) - np.mean(common_y))**2)
    var_sy = (y.var(ddof=1)
              + w1 * (gamma_2**2) * (common_x.var(ddof=1) - common_y.var(ddof=1))
              + w1 * w2 * (gamma_2**2) * (np.mean(common_x) - np.mean(common_y))**2)

    #Standard deviations
    sd_sx = np.sqrt(var_sx)
    sd_sy = np.sqrt(var_sy)

    if sd_sx == 0:
        raise ValueError("Synthetic SD of X is zero; cannot compute slope.")

    scores = np.asarray(scores, dtype=float)
    ly_x = (sd_sy / sd_sx) * (scores - mu_sx) + mu_sy

    eyx = pd.DataFrame({'Scores': scores,
                        'ey': ly_x})
    return eyx
