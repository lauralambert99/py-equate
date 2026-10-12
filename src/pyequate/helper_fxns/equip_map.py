# -*- coding: utf-8 -*-
"""
Shared equipercentile mapping used by fe() and chained().

@author: laycocla
"""

import numpy as np


def percentile_rank(freq, points):
    """
    Continuous percentile rank, as a proportion (Kolen & Brennan, 2004, Eq. 2.14).

    Parameters
    ----------
    freq : pd.Series
        Frequencies (or probabilities) indexed by ascending integer score points.
    points : array-like
        Score values, integer or not, at which to evaluate the percentile rank.
        Values outside [min - 0.5, max + 0.5] are clamped to 0 or 1.

    Returns
    -------
    np.ndarray
        Percentile rank of each point, between 0 and 1.
    """
    scores = freq.index.to_numpy(dtype=float)
    F = (freq / freq.sum()).cumsum().to_numpy(dtype=float)

    #Piecewise linear, with knots at each score - 0.5 and at the top score + 0.5
    knots = np.append(scores - 0.5, scores[-1] + 0.5)
    values = np.append(0.0, F)
    return np.interp(np.asarray(points, dtype=float), knots, values)


def inverse_percentile_rank(freq, p):
    """
    Score with percentile rank p on the scale of `freq` (Kolen & Brennan, 2004, Eq. 2.17).

    Parameters
    ----------
    freq : pd.Series
        Frequencies (or probabilities) indexed by ascending integer score points.
    p : array-like
        Percentile ranks, as proportions between 0 and 1.

    Returns
    -------
    np.ndarray
        Score on the `freq` scale for each entry of `p`.
    """
    G = (freq / freq.sum()).cumsum().to_numpy(dtype=float)
    to_scores = freq.index.to_numpy(dtype=float)

    p = np.asarray(p, dtype=float)
    out = np.empty(len(p))
    for i, p_i in enumerate(p):
        #Smallest target score whose cumulative proportion reaches p (Y*_U)
        k = min(int(np.searchsorted(G, p_i, side="left")), len(G) - 1)
        Ghi = G[k]
        Glo = G[k - 1] if k > 0 else 0.0
        if (Ghi <= Glo) or np.isclose(Ghi, Glo):
            out[i] = to_scores[k]
        else:
            #Linear interpolation within the target bin, with 0.5 continuity correction
            out[i] = (to_scores[k] - 0.5) + (p_i - Glo) / (Ghi - Glo)
    return out


def equip_map(from_freq, to_freq):
    """
    Equipercentile map from one score scale to another (Kolen & Brennan, 2004, Eq. 2.17).

    Parameters
    ----------
    from_freq : pd.Series
        Frequencies (or probabilities) of the form being equated, indexed by
        ascending score points.
    to_freq : pd.Series
        Frequencies (or probabilities) of the target form, indexed by
        ascending score points. May use a different score range than `from_freq`.

    Returns
    -------
    np.ndarray
        Equated score on the `to_freq` scale for each score point of `from_freq`.
    """
    f = from_freq / from_freq.sum()

    #Percentile ranks (as proportions) of the from-scores
    P = (f.cumsum().shift(1, fill_value=0) + f / 2).to_numpy(dtype=float)
    return inverse_percentile_rank(to_freq, P)
