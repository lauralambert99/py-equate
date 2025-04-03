# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:00:57 2025

@author: Laura
"""

def neat(x, y, score_min, score_max, type="l"):
    """
    A function to perform single group equating.

    Parameters:
    x : array of scores on Form X
    y : array of scores on Form Y 
    score_min: minimum score on the form
    score_max: maximum score on the form
    type : str, optional
        Type of equating; "l" (linear methods) and "eq" (equipercentile methods) are accepted. 
        Default is "l".


    Returns:
    DataFrame
        A DataFrame containing yx values for equated x scores
    """
