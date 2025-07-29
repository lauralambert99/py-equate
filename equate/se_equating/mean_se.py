# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:07:56 2025

@author: Laura
"""
import numpy as np

def mean_se(formX, formY):
    """
    Compute the standard error of equating (SEE) for mean equating
    in a random groups design using raw scores and the Delta Method

    Parameters:
    - formX: array-like raw scores from Form X
    - formY: array-like raw scores from Form Y

    Returns:
    - standard error of equating (float)
    """
    formX = np.asarray(formX)
    formY = np.asarray(formY)

    var_x = np.var(formX, ddof=1)
    var_y = np.var(formY, ddof=1)

    n_x = len(formX)
    n_y = len(formY)

    se2 = var_x / n_x + var_y / n_y
    
    return np.sqrt(se2)

