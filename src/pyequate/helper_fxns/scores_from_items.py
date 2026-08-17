# -*- coding: utf-8 -*-
"""
Helper function to construct total and anchor scores from item-level response data.

Intended location: helper_fxns/scores_from_items.py
(then add to helper_fxns imports in the package __init__.py, e.g.,
 `from .helper_fxns.scores_from_items import scores_from_items`)

Constructing the scores inside the package removes the most common
user error in the NEAT design: with an internal anchor, the total
score MUST include the common items; with an external anchor, it
must NOT. When users build totals by hand it is easy to pass the
wrong kind of total, which silently corrupts the gamma terms in the
Levine (and Tucker) methods.

@author: laycocla
"""

import numpy as np
import pandas as pd


def scores_from_items(responses, common, anchor="internal"):
    """
    Build total and anchor score vectors from item-level responses.

    Parameters
    ----------
    responses : pandas.DataFrame or 2D array-like
        Item response matrix (rows = examinees, columns = items).
        Entries are item scores (e.g., 0/1 for dichotomous items).
    common : array-like
        Identifies the common (anchor) items. If `responses` is a
        DataFrame, `common` may be a list of column NAMES or a list of
        integer positions; for an array, integer positions only.
    anchor : str, optional
        "internal": anchor items are part of the test, so the total
                    score is the sum over ALL items (anchor included).
        "external": anchor is scored separately, so the total score is
                    the sum over the NON-common items only.
        Defaults to "internal".

    Returns
    -------
    total : pandas.Series
        Total score per examinee, defined according to `anchor`.
    anchor_score : pandas.Series
        Anchor (common-item) score per examinee.

    Notes
    -----
    Rows with any missing item response are dropped (listwise), with a
    warning, so that total and anchor scores stay paired examinee-wise.
    """
    valid_anchor = ["internal", "external"]
    if anchor not in valid_anchor:
        raise ValueError(f"Anchor '{anchor}' not supported. Choose from {valid_anchor}")

    responses = pd.DataFrame(responses).copy()

    # Resolve `common` to column labels
    common = list(common)
    if all(c in responses.columns for c in common):
        common_cols = common
    elif all(isinstance(c, (int, np.integer)) for c in common):
        try:
            common_cols = list(responses.columns[common])
        except IndexError:
            raise ValueError("Integer positions in `common` are out of range "
                             f"for a response matrix with {responses.shape[1]} items.")
    else:
        missing = [c for c in common if c not in responses.columns]
        raise ValueError(f"Common items not found in response matrix: {missing}")

    if len(set(common_cols)) != len(common_cols):
        raise ValueError("`common` contains duplicate items.")
    if len(common_cols) == responses.shape[1]:
        raise ValueError("All items are marked common; no unique items remain.")

    # Listwise deletion of incomplete rows to keep scores paired
    n_before = len(responses)
    responses = responses.dropna()
    if len(responses) < n_before:
        import warnings
        warnings.warn(f"Dropped {n_before - len(responses)} examinee(s) with "
                      "missing item responses (listwise deletion).")

    noncommon_cols = [c for c in responses.columns if c not in common_cols]

    anchor_score = responses[common_cols].sum(axis=1)

    if anchor == "internal":
        # Total includes the anchor items
        total = responses.sum(axis=1)
    else:
        # External anchor: total excludes the separately scored anchor
        total = responses[noncommon_cols].sum(axis=1)

    return total, anchor_score