    
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:07:55 2025

@author: Laura
"""

#Load in dependent packages
import pandas as pd
import numpy as np

#%%

def equipercen(x, y, score_min, score_max):
    """
    A function to create a frequency table for equating from two score vectors
    
    Parameters
    ----------
    x: a vector or list of scores on the OLD form
    y: a vector or list of scores on the NEW form
    score_min: the minimum possible score
    score_max: the maximum possible score
    
    Returns: a frequency table
    
    Depends: pandas, itertools
    
    TO-DO: Flexibility!
    TO-DO: Pop out little embedded functions into their own things?

    """
    #TODO: potential errors
        #If missing values for certain cells, may run into issues
        #What to do with missing values
    #TODO: Testing function works as expected
    
    #Convert to series just in case?
    x = pd.Series(x)
    y = pd.Series(y)
    
    #First, get values in a table
    pfreq_x = x.value_counts().reindex(range(score_min, score_max + 1), fill_value=0).sort_index()
    pfreq_y = y.value_counts().reindex(range(score_min, score_max + 1), fill_value=0).sort_index()

    
    #Calculate f(x) and g(y)
    f_x = pfreq_x/pfreq_x.sum()
    g_y = pfreq_y/pfreq_y.sum()
    
    #Calculate F(x) and G(y)
    Fx = f_x.cumsum()
    Gy = g_y.cumsum()

    Px = 100*(Fx.shift(1, fill_value = 0) + f_x/2)
    
    #Make G(y) * 100 value column - easier to reference this
    Gy_100 = Gy*100

    #Make a function to take each P(x) value and find the smallest Gy_100 value that is => to it

    #But what we really want is the corresponding Y value
    #Edit above to give us that
    def find_Y_star(Px, Gy, Y):
      idx = np.searchsorted(Gy, Px, side="left")
      return Y[idx] if idx < len(Y) else None

    scores = np.arange(score_min, score_max + 1)
    pdata = pd.DataFrame({
        'Score': scores, 
        'X': pfreq_x.values, 
        'Y': pfreq_y.values, 
        'f_x': f_x.values, 
        'g_y': g_y.values, 
        'Fx': Fx.values, 
        'Gy': Gy.values,
        'Px': Px.values,
        'Gy_100': Gy_100.values
    })

    #Use the function to make a new column 
    pdata['Y_star_u'] = pdata['Px'].apply(lambda Px: find_Y_star(Px, pdata['Gy_100'], pdata['Score']))

    #Compute G(Y*u)
    pdata['Gy_star'] = pdata['Y_star_u'].apply(lambda y_star: Gy[y_star] if pd.notna(y_star) else None)
    
    #Will also need a lag Y*u for equation
    pdata['Gy_star_lag'] = pdata['Y_star_u'].apply(lambda y_star: Gy[y_star - 1] if pd.notna(y_star) and y_star > score_min else None)

    # Compute equated scores
    pdata['e_yx'] = ((((pdata['Px'] / 100) - pdata['Gy_star_lag']) / 
                      (pdata['Gy_star'] - pdata['Gy_star_lag'])) + 
                      (pdata['Y_star_u'] - 0.5))

    return {'yx': pdata['e_yx']}


#%%
old = ADM2['x']
new = ADM1['x']

#Testing - works in this instance
equipercen(old, new, 0, 50)
    
    