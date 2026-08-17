# -*- coding: utf-8 -*-
"""
Levine True Score equating

Implements the Levine true score linear equating function under the
classical congeneric model (Kolen & Brennan, 2004):

    l_Y(x) = (gamma_2 / gamma_1) * (x - mu_1(X))
             + mu_2(Y)
             + gamma_2 * (mu_1(V) - mu_2(V))

Accepts either total-score input (x, y, common_x, common_y) or
item-level input (items_x, items_y, common), in which case totals and
anchor scores are constructed internally via scores_from_items().

@author: laycocla
"""

import numpy as np
import pandas as pd

from ..helper_fxns.scores_from_items import scores_from_items

def LevineTS(x=None, y=None, common_x=None, common_y=None, scores=None, w1=None,
             anchor="internal", items_x=None, items_y=None, common=None):
    """
    Perform Levine True Score equating.

    Provide EITHER total scores OR item-level responses:

    Total-score input
    -----------------
    x, y : pandas.Series
        Raw total scores for Form X (population 1) and Form Y
        (population 2). With anchor="internal" these totals must
        INCLUDE the common items; with anchor="external" they must not.
    common_x, common_y : pandas.Series
        Anchor scores, paired row-wise with x and y respectively.

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
        IGNORED: Levine true score equating is invariant to the
        synthetic weights. Retained only so the signature matches
        Tucker() and LevineOS().
    anchor : str, optional
        "internal" or "external". Defaults to "internal".

    Returns
    -------
    pandas.DataFrame
        Columns 'Scores' (Form X score points) and 'ey' (Levine true
        score equivalents on the Form Y scale).

    Notes
    -----
    Although the transformation is derived for true scores, in practice
    it is applied to observed scores (Kolen & Brennan, 2004, Sec. 4.2.3).
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

    #Transformation
    mu_x = x.mean()
    mu_y = y.mean()
    mu_vx = common_x.mean()   # mu_1(V): anchor mean in population 1
    mu_vy = common_y.mean()   # mu_2(V): anchor mean in population 2

    slope = gamma_2 / gamma_1
    scores = np.asarray(scores, dtype=float)

    ly_x = slope * (scores - mu_x) + mu_y + gamma_2 * (mu_vx - mu_vy)

    eyx = pd.DataFrame({'Scores': scores,
                        'ey': ly_x})
    return eyx
