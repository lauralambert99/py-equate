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
ADM1 = pd.read_csv(r'C:\Users\laycocla\OneDrive - James Madison University\Documents\A&M\Equating\Homework_1\form_y.csv')
ADM2 = pd.read_csv(r'C:\Users\laycocla\OneDrive - James Madison University\Documents\A&M\Equating\Homework_1\form_x.csv')


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
#NEEDS WORK - SEE TEST BELOW
def freqtab(data, scales = None, items = None):
    """
    A function to create a frequency table for equating from a dataframe
    Parameters
    ----------
    x : a pd.dataframe with total scores
    scales : provides the measurement scales for the specified variables
    items: which columns are to be used in the frequency table
    
    Returns: a frequency table

    """
    #Make sure it's a dataframe
    data = pd.DataFrame(data)
      
    #Get number of columns in the data
    nx = data.shape[1]
    
    #If scales = none, compute scales from data (it would run min to max for each column)
    if scales is None:
        scales = [list(range(int(data[col].min()), int(data[col].max()) + 1)) for col in data.columns]
    
    #Ensure scales is a list of lists
    if not isinstance(scales, list):
        scales = [scales]

    # Convert each column to categorical with the provided or computed scales
    for i in range(nx):
        data.iloc[:, i] = pd.Categorical(data.iloc[:, i], categories=scales[i], ordered=True)
    
    #Check if items are provided
    if items is not None:
        if not isinstance(items, list):
            items = [items]
        
        #Row sums for selected items
        data = pd.DataFrame({i: data[i].sum(axis=1) if isinstance(data[i], pd.DataFrame) else data[i] 
                             for i in items})
        
    # Create a frequency table (similar to 'table' in R)
    freq_table = data.apply(pd.Series.value_counts).fillna(0)
    
    
    return freq_table

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

#Scores as a dataframe
test = pd.DataFrame({'X': ADM2['x'], 'Y': ADM1['x']})

#Define the full range of possible scores
#This only counts scores that were observed - no zeros for unobserved combinations
formx_range = range(int(test['X'].min()), int(test['X'].max()) + 1)  #Need the plus one because not inclusive
formy_range = range(int(test['Y'].min()), int(test['Y'].max()) + 1)  

#Create a grid of all different observed score combinations
all_combinations = pd.DataFrame(itertools.product(formx_range, formy_range), columns=['X', 'Y'])

#Get frequency counts
freq_table = test.groupby(['X', 'Y']).size().reset_index(name='count')

#Merge with all possible observed combinations, filling missing counts with 0
full_freq_table = all_combinations.merge(freq_table, on=['X', 'Y'], how='left').fillna(0)

#Make sure 'count' is integer
full_freq_table['count'] = full_freq_table['count'].astype(int)

#Rename columns to something I understand
full_freq_table.columns = ['X', 'Y', 'count']



#Test mean and linear equating using function
lx = linear(full_freq_table['X'], full_freq_table['Y'])

mx = linear(full_freq_table['X'], full_freq_table['Y'], type = "mean")



