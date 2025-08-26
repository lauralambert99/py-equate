# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:54:01 2025

@author: Laura
"""

import pandas as pd
import numpy as np
import itertools


def freqtab(data2, data1):
    """
    A function to create a frequency table for equating from two score vectors
    
    Parameters
    ----------
    data1: a vector or list of scores on the OLD form
    data2: a vector or list of scores on the NEW form
    
    Returns: a frequency table
    
    Depends: pandas, itertools

    """
    #Make sure it's a dataframe
    data = pd.DataFrame({'X': data2, 'Y': data1})
     
    #Get the range of each form
    formx_range = range(int(data['X'].min()), int(data['X'].max()) + 1)  #Need the plus one because not inclusive
    formy_range = range(int(data['Y'].min()), int(data['Y'].max()) + 1)  

    #Create a grid of all different observed score combinations
    all_combinations = pd.DataFrame(itertools.product(formx_range, formy_range), columns=['X', 'Y'])

    #Get frequency counts
    freq_table = data.groupby(['X', 'Y']).size().reset_index(name='count')

    #Merge with all possible observed combinations, filling missing counts with 0
    full_freq_table = all_combinations.merge(freq_table, on=['X', 'Y'], how='left').fillna(0)

    #Make sure 'count' is integer
    full_freq_table['count'] = full_freq_table['count'].astype(int)

    #Rename columns to something understandable
    full_freq_table.columns = ['X', 'Y', 'count']
  
    
    return full_freq_table

def linear(x, y, score_min, score_max, type="linear", rescale=False, group="Random"):
    """
    A function to perform mean and linear equating.

    Parameters:
    x : array of new scores
    y : array of old scores 
    score_min: minimum score on the form
    score_max: maximum score on the form
    type : str, optional
        Type of equating. "mean" (mean equating),  "linear" (linear equating), "zscore" (z-score equating) are accepted
        Default is "linear".
    rescale : bool, optional
        Whether to rescale scores to a 0-100 range. Default is False.
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
            
    if rescale:
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 100
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 100
    
    #Define scores
    scores = np.arange(score_min, score_max + 1)
    
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
    elif type == "zscore":
        slope = 1
        intercept = 0
        x = (x - mean_x) / np.std(x)
        y = (y - mean_y) / np.std(y)
    else:
        raise ValueError("Invalid 'type'. Use 'mean', 'linear', or 'zscore'.")
    
    ex = slope * scores + intercept
    
    eq =  pd.DataFrame({'Score': scores,
                        'ex': ex})
    #Return equated scores as a dataframe
    return eq



