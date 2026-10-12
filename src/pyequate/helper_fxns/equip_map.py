# -*- coding: utf-8 -*-
"""
Shared equipercentile mapping used by fe() and chained().

@author: laycocla
"""

import numpy as np


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
    g = to_freq / to_freq.sum()

    #Percentile ranks (as proportions) of the from-scores, and target cumulative distribution
    P = (f.cumsum().shift(1, fill_value=0) + f / 2).to_numpy(dtype=float)
    G = g.cumsum().to_numpy(dtype=float)
    to_scores = to_freq.index.to_numpy(dtype=float)

    out = np.empty(len(P))
    for i, p in enumerate(P):
        #Smallest target score whose cumulative proportion reaches p (Y*_U)
        k = min(int(np.searchsorted(G, p, side="left")), len(G) - 1)
        Ghi = G[k]
        Glo = G[k - 1] if k > 0 else 0.0
        if (Ghi <= Glo) or np.isclose(Ghi, Glo):
            out[i] = to_scores[k]
        else:
            #Linear interpolation within the target bin, with 0.5 continuity correction
            out[i] = (to_scores[k] - 0.5) + (p - Glo) / (Ghi - Glo)
    return out
