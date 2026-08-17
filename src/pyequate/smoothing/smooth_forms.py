# -*- coding: utf-8 -*-
"""
Bridge between :func:`presmooth` and the equipercentile equating functions.

``presmooth`` smooths a single form and returns a wide table (one column per
polynomial order).  ``equipercen`` and ``loop_equate`` instead expect a single
long DataFrame holding *both* forms, with columns
``['form', 'order', 'score', 'smoothed_freq']``.  ``smooth_forms`` runs the
presmoothing for each form and reshapes the results into that long format.
"""

from __future__ import annotations

import pandas as pd

from .presmooth import presmooth

__all__ = ["smooth_forms"]


def _one_form_long(freq, form, score_min, score_max, degrees):
    """Presmooth one form and return it in long format."""
    max_degree = int(degrees[0])
    wide = presmooth(freq, score_min, score_max, degrees=degrees, stepup=True)
    frames = []
    for order, col in enumerate(wide.columns, start=1):
        frames.append(
            pd.DataFrame(
                {
                    "form": form,
                    "order": order,
                    "score": wide.index.to_numpy(),
                    "smoothed_freq": wide[col].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def smooth_forms(x, y, score_min, score_max, degrees=None):
    """
    Presmooth two score forms and return frequencies in the long format used
    by :func:`equipercen` and :func:`loop_equate`.

    Parameters
    ----------
    x : array-like
        Observed score frequencies for the X (OLD) form, ordered from
        ``score_min`` to ``score_max`` (or a Series/dict indexed by score).
    y : array-like
        Observed score frequencies for the Y (NEW) form.
    score_min : int
        Minimum possible score.
    score_max : int
        Maximum possible score.
    degrees : list of int, optional
        Maximum polynomial degree, as a single-element list (e.g. ``[4]``).
        Models of order ``1 .. degrees[0]`` are fit for each form. Defaults
        to ``[4]``.

    Returns
    -------
    DataFrame
        Long-format presmoothed frequencies with columns
        ``['form', 'order', 'score', 'smoothed_freq']``, where ``form`` is
        ``'X'`` or ``'Y'`` and ``order`` runs from 1 to ``degrees[0]``. Pass
        the result to :func:`equipercen` (with an ``order``) or
        :func:`loop_equate`.
    """
    if degrees is None:
        degrees = [4]

    x_long = _one_form_long(x, "X", score_min, score_max, degrees)
    y_long = _one_form_long(y, "Y", score_min, score_max, degrees)
    return pd.concat([x_long, y_long], ignore_index=True)
