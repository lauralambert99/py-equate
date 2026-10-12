# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:54:01 2025

@author: Laura
"""

import pandas as pd
import numpy as np

def linear(x, y, score_min, score_max, type="linear", rescale=False, group="Random"):
    """
    A function to perform mean and linear equating.

    Parameters:
    x : array of new scores
    y : array of old scores 
    score_min: minimum score on the form
    score_max: maximum score on the form
    type : str, optional
        Type of equating. "mean" (mean equating) or "linear" (linear equating).
        Default is "linear". (z-score equating, setting z_X(x) = z_Y(y), is linear equating.)
    rescale : bool, optional
        If True, x, y and the score points are rescaled to 0-100 using the possible
        score range (score_min..score_max), and 'ex' is reported on that 0-100 scale.
        'Score' stays on the raw scale. Default is False.
    group: str, optional
        Type of group method of equating.  "Random" (Random groups design) and "Single" are accepted.
        Default is "Random".

    Returns:
    DataFrame
        A DataFrame containing yx values for equated x scores
    """
    #TODO: potential errors
        #Negative score numbers
        #Scores need to be numbers
        #Maybe drop NAs
        #SD of zero would be an issue!
    #TODO: Testing function works as expected
        
    #This should be the only score consideration, right?
    if group == "Single":
        #Make sure scores are paired for single group design
        if len(x) != len(y):
            raise ValueError("X and Y must have the same length for single-group equating (paired scores).")

    #Define scores
    scores = np.arange(score_min, score_max + 1)
    eq_points = scores.astype(float)

    if rescale:
        #Same transform for both forms and the score points, based on the possible score range
        span = score_max - score_min
        x = (np.asarray(x, dtype=float) - score_min) / span * 100
        y = (np.asarray(y, dtype=float) - score_min) / span * 100
        eq_points = (eq_points - score_min) / span * 100

    #Compute the means of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if type == "mean":
        #Mean equating: align the means of x and y
        intercept = mean_y - mean_x
        slope = 1  #No change in slope with mean equating, just intercepts
    elif type == "linear":
        #Linear equating: align both the means and standard deviations
        sd_x = np.std(x)  #Standard deviation of x
        sd_y = np.std(y)  #Standard deviation of y
        slope = sd_y / sd_x  #Ratio of standard deviations
        intercept = mean_y - slope * mean_x
    else:
        raise ValueError("Invalid 'type'. Use 'mean' or 'linear'.")

    ex = slope * eq_points + intercept

    eq =  pd.DataFrame({'Score': scores,
                        'ex': ex})
    #Return equated scores as a dataframe
    return eq



