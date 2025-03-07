# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:54:01 2025

@author: Laura
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
#%%

#Get data
ADM1 = pd.read_csv(r'C:\Users\Laura\OneDrive - James Madison University\Documents\A&M\Equating\Homework_1\form_y.csv')
ADM2 = pd.read_csv(r'C:\Users\Laura\OneDrive - James Madison University\Documents\A&M\Equating\Homework_1\form_x.csv')

#Vectors of columns
form_x = ADM2['x']
form_y = ADM1['x']

#First, get in a table
pfreq_x = ADM2['x'].value_counts().sort_index()
pfreq_y = ADM1['x'].value_counts().sort_index()

#Python likes it to be in a dictionary before matching up?
#That's what the internet says, anyway.
dict_x = pfreq_x.to_dict()
dict_y = pfreq_y.to_dict()

#Make a vector of scores
#Remember python is a 0 index, and doesn't include the last number
scores = np.arange(0, 51)

#Make empty arrays
array_x = np.zeros(len(scores), dtype = int)
array_y = np.zeros(len(scores), dtype = int)

#Fill the arrays
for key, value in dict_x.items():
  array_x[key] = value
  
for key, value in dict_y.items():
  array_y[key] = value

#Combine into a dataframe
pdata = pd.DataFrame({'Score': scores, 'X': array_x, 'Y': array_y})


#%%

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

def linear(x, y, type="linear"):
    """
    A function to perform mean and linear equating.

    Parameters:
    x : array of new scores
    y : array of old scores 
    type : str, optional
        Type of equating. Either "mean" (mean equating) or "linear" (linear equating) are accepted
        Default is "linear".

    Returns:
    dict
        A dictionary containing yx values for equated x scores
    """

    #Compute the means of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if type == "mean":
        #Mean equating: align the means of x and y
        intercept = mean_y - mean_x
        slope = 1  #No change in slope with mean equating, just intercepts
        yx = x + intercept
    elif type == "linear":
        #Linear equating: align both the means and standard deviations
        sd_x = np.std(x)  #Standard deviation of x
        sd_y = np.std(y)  #Standard deviation of y
        slope = sd_y / sd_x  #Ratio of standard deviations
        intercept = mean_y - slope * mean_x  
        yx = slope*x + intercept

    # Return the slope and intercept as a dictionary
    return {'yx':yx}

#%%
#Testing
#Freqtab first
data = freqtab(form_x, form_y)

#Test mean and linear equating using function
lx = linear(data['X'], data['Y'])

mx = linear(data['X'], data['Y'], type = "mean")



