    
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:07:55 2025

@author: Laura
"""

#Load in dependent packages
import numpy as np
import pandas as pd

from typing import Literal

#%%

# ---------------------------------------------------------------------------
# Find Y* function
# ---------------------------------------------------------------------------
 
def _find_Y_star(px_val: float, Gy_arr: np.ndarray) -> int:
    """
    Return the index k such that Gy_arr[k] is the first value >= px_val.
    Clipped to [0, len(Gy_arr) - 1].
 
    Parameters
    ----------
    px_val : float
        Percentile rank of an X score (0–100 scale).
    Gy_arr : np.ndarray
        Cumulative distribution of Y on a 0–100 scale.
 
    Returns
    -------
    k : int
        Index into Gy_arr / scores array.
    """
    k = int(np.searchsorted(Gy_arr, px_val, side="left"))
    return min(k, len(Gy_arr) - 1)

# ---------------------------------------------------------------------------
# Core single-pair equating function
# ---------------------------------------------------------------------------
#Pull workhorse function out, call it in overall function
def _single_equip(xfreq: pd.Series, yfreq: pd.Series) -> pd.DataFrame:
    """
    Parameters
    ----------
    xfreq : pd.Series
        Frequency distribution for Form X, indexed by integer scores.
    yfreq : pd.Series
        Frequency distribution for Form Y, indexed by integer scores.

    Returns
    -------
    pd.DataFrame with columns ['score', 'equated'].
    """
    scores = xfreq.index.to_numpy(dtype = float)
    
    #Convert to probabilities
    f_x = xfreq / xfreq.sum()
    g_y = yfreq / yfreq.sum()

    #Cumulative distributions
    Fx = f_x.cumsum()
    Gy = g_y.cumsum()
    
    #Percentile ranks for X
    Px = 100 * (Fx.shift(1, fill_value=0) + f_x / 2) #Give cumulative proportion below score X
    
    Gy_100 = (Gy * 100).to_numpy()
    Gy_arr = Gy.to_numpy()
    Px_arr = Px.to_numpy()

    #Interpolated equated scores
    equated = []
    for px in Px_arr:
        k = _find_Y_star(px, Gy_100)        #positional index in arrays
        Ghi = float(Gy_arr[k])                   #positional lookup
        Glo = float(Gy_arr[k-1]) if k > 0 else 0.0
        
        if (Ghi <= Glo) or np.isclose(Ghi, Glo):
            yhat = float(scores[k])
        else:
            #Linear interpolation within the Y bin, with 0.5 continuity correction
            yhat = (float(scores[k]) - 0.5) + ((px/100.0 - Glo) / (Ghi - Glo))
            
        equated.append(float(yhat))

    equated_df = pd.DataFrame({'score': scores, 'equated': equated})
    
    return equated_df

# ---------------------------------------------------------------------------
# Presmooth conversion helper functions
# ---------------------------------------------------------------------------
#Handle different types of output from presmooth()

def _freq_from_smoothed(
    result,
    score_min: int,
    score_max: int,
    degree: int | str | None,) -> pd.Series:
    """
    Extract a single frequency Series from presmooth() output, regardless of type

    Parameters
    ----------
    result : pd.Series | pd.DataFrame | dict
        Output of presmooth().
    score_min, score_max : int
        Used to validate index alignment.
    degree : int, str, or None
        Which model to select when result is a stepup DataFrame or choose dict.
        Pass an integer (e.g. 4) or a column name (e.g. "Degree 4").
        Ignored when result is already a plain Series.

    Returns
    -------
    pd.Series indexed by integer scores.
    """
    expected_index = pd.Index(range(score_min, score_max + 1), name="score")

    #Figure out what type of result we're working with
    #Option 1: Dictionary
    if isinstance(result, dict):
        if "fitted" not in result:
            raise ValueError(
                "dict result from presmooth() must contain a 'fitted' key. "
                "Pass choose = True when calling presmooth().")
        if degree is not None:
            raise ValueError(
                "To select a specific degree from multiple models, pass the "
                "stepup DataFrame (presmooth(..., stepup = True)) rather than "
                "the choose dict.")
        
        series = result["fitted"]
        
    #Option 2: DataFrame from stepup
    elif isinstance(result, pd.DataFrame):
        if degree is None:
            raise ValueError(
                "Must specify 'degree' when passing a stepup DataFrame from "
                "presmooth(..., stepup = True). "
                "Pass an integer (e.g., degree = 4) or column nam (e.g. 'Degree 4').")
        col = f"Degree {degree}" if isinstance(degree, int) else degree
        if col not in result.columns:
            raise ValueError(
                f"Column '{col}' not found in stepup DataFrame. "
                f"Available: {list(result.columns)}")
        
        series = result[col]
    
    #Option 3: Series
    elif isinstance(result, pd.Series):
        series = result
    else:
        raise TypeError(
            f"Unrecognised presmooth() output type: {type(result)}. "
            "Expected pd.Series, pd.DataFrame, or dict.")
        
    if not series.index.equals(expected_index):
        series = series.reindex(expected_index, fill_value = 0)
        
    return series.astype(float)

#Also need a function to turn raw score vectors into frequencies
def _freq_from_raw(
        scores: "array-like",
        score_min: int,
        score_max: int,) -> pd.Series:
    
    """
    Parameters
    ----------
    scores: array-like of numeric scores
    score_min, score_max: int
    
    Returns
    -------
    pd.Series indexed by integer scores (score_min to score_max)
    """
    s = pd.Series(scores)
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError("Raw score vector must be numeric.")
        
    out_of_range = s[(s < score_min) | (s > score_max)]
    if len(out_of_range) > 0:
        raise ValueError(
            f"{len(out_of_range)} score(s) fall outside "
            f"[{score_min}, {score_max}] and would be dropped. "
            "Check score_min / score_max.")
    
    return(
        s.value_counts()
         .reindex(range(score_min, score_max + 1), fill_value = 0)
         .sort_index()
         .rename_axis("score")
         .astype(float))

# ---------------------------------------------------------------------------
# Overall Equipercentile function
# ---------------------------------------------------------------------------
#This is what users will call    

def equipercen(score_min: int, 
               score_max: int,
               x: "array-like | None" = None,
               y: "array-like | None" = None,
               x_smoothed = None,
               y_smoothed = None,
               x_degree: "int | str | None" = None,
               y_degree: "int | str | None" = None,) -> pd.DataFrame:
    """
    A function to perform random groups equipercentile equating.
    
    Can accept raw score vectors x, y or output from presmooth() as pd.Series, pd.DataFrame, or dict.,
    or a combination of raw scores/presmoothed scores
    
    Parameters
    ----------
    score_min: int
        Minimum possible score on both forms.
    score_max: int
        Maximum possible score on both forms.
    x: array-like, optional
        Raw score vector for Form X (the old form being equated FROM).
    y: array-like, optional
        Raw score vector for Form Y (the new form being equated TO).
    x_smoothed: pd.Series | pd.DataFrame | dict, optional
        Direct output of loglinear() for Form X.
    y_smoothed: pd.Series | pd.DataFrame | dict, optional
        Direct output of loglinear() for Form Y.
    x_degree: int or str, optional
        Degree to select when x_loglinear is a stepup DataFrame,
        e.g. 4 or "Degree 4". Ignored for Series / choose-dict inputs.
    y_degree: int or str, optional
        Same as x_degree but for Form Y.

    Returns
    -------
    "equated"  : pd.DataFrame — columns ['score', 'equated']
    
    """
    
    # ------------------------------------------------------------------
    # Initial check: make sure exactly one source is provided per form
    # ------------------------------------------------------------------
    if (x is None) == (x_smoothed is None):
        raise ValueError(
            "Provide exactly one of 'x' (raw vector) or "
            "'x_presmooth' (presmooth output) for Form X.")
    if (y is None) == (y_smoothed is None):
        raise ValueError(
            "Provide exactly one of 'y' (raw vector) or "
            "'y_presmooth' (presmooth output) for Form Y.")
    
    # ------------------------------------------------------------------
    # Make sure each form is a frequency
    # ------------------------------------------------------------------
    if x_smoothed is not None:
        xfreq = _freq_from_smoothed(x_smoothed, score_min, score_max, x_degree)
    else:
        xfreq = _freq_from_raw(x, score_min, score_max)
    
    if y_smoothed is not None:
        yfreq = _freq_from_smoothed(y_smoothed, score_min, score_max, y_degree)
    else:
        yfreq = _freq_from_raw(y, score_min, score_max)
    
    # ------------------------------------------------------------------
    # Do equating!
    # ------------------------------------------------------------------
    equated_df = _single_equip(xfreq, yfreq)
    
    out = equated_df
    
    return out
    
    
    
    
    