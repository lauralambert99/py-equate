# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:07:55 2025

@author: Laura
"""

#Load in dependent packages
import pandas as pd
import numpy as np

#%%

def equipercen_original(x, y, score_min, score_max):
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
    TO-DO: Probably don't need everything to be in a data frame
    TO-DO: Pop out little embedded functions into their own things?

    """
    #Convert to series just in case?
    x = pd.Series(x)
    y = pd.Series(y)
    
    #First, get values in a table
    pfreq_x = x.value_counts().sort_index()
    pfreq_y = y.value_counts().sort_index()


    #Put in a dictionary
    dict_x = pfreq_x.to_dict()
    dict_y = pfreq_y.to_dict()

    #Make a vector of scores
    #Remember python is a 0 index, and doesn't include the last number - CHECK THAT SCORES IS MADE RIGHT
    scores = np.arange(score_min, (score_max + 1))

    #Make empty arrays
    array_x = np.zeros(len(scores), dtype = int)
    array_y = np.zeros(len(scores), dtype = int)

    #Fill the arrays
    for key, value in dict_x.items():
      array_x[key] = value
      
    for key, value in dict_y.items():
      array_y[key] = value

    #Combine into a dataframe
    #Do we actually need it to go into a dataframe??
    pdata = pd.DataFrame({'Score': scores, 'X': array_x, 'Y': array_y})

    
    #Calculate f(x) and g(y)
    pdata['f_x'] = pdata['X']/sum(pdata['X'])
    pdata['g_y'] = pdata['Y']/sum(pdata['Y'])
    
    #Calculate F(x) and G(y)
    pdata['Fx'] = np.cumsum(pdata['f_x'])
    pdata['Gy'] = np.cumsum(pdata['g_y'])

    pdata['Px'] = 100*(pdata['Fx'].shift(1) + (pdata['f_x']/2))
    pdata['Qy'] = 100*(pdata['Gy'].shift(1) + (pdata['g_y']/2))
    
    #Make G(y) * 100 value column - easier to reference this
    pdata['Gy_100'] = pdata['Gy']*100

    #Make a function to take each P(x) value and find the smallest Gy_100 value that is => to it
    #Should these be popped out into their own functions??

    #Started by finding  smallest GY_100 >= Px
    def find_gte_Gy(Px, Gy100):
      Gy_gte = Gy100[Gy100 >= Px] #First, get values
      
      if len(Gy_gte) > 0:
        return Gy_gte.min()
      else:
        return None

    pdata['min_Gy'] = pdata['Px'].apply(lambda Px: find_gte_Gy(Px, pdata['Gy_100']))

    #But what we really want is the corresponding Y value
    #Edit above to give us that
    def find_Y_star(Px, Gy, Y):
      gte = np.where(Gy >= Px)[0]
      
      if len(gte) > 0:
        Gy_index = gte[np.argmin(Gy[gte])]
        return Y[Gy_index]
      else:
        return None

    #Use the function to make a new column 
    pdata['Y_star_u'] = pdata['Px'].apply(lambda Px: find_Y_star(Px, pdata['Gy_100'], pdata['Score']))

    #Make a G(Y*u) column
    def find_GY_star(Y_star, Y, Gy):
      
      try:
        get_index = Y.index(Y_star)
        return Gy[get_index]
      except ValueError:
        return None

    #Make the GY* column
    pdata['Gy_star'] = pdata['Y_star_u'].apply(lambda Y_star_u: find_GY_star(Y_star_u, pdata['Score'].tolist(), pdata['Gy'].tolist()))


    #Will also need a lag Y*u for equation
    def find_GY_star_lag(Y_star, Y, Gy):
      
      try:
        get_index = Y.index(Y_star - 1)
        return Gy[get_index]
      except ValueError:
        return None

    pdata['Gy_star_lag'] = pdata['Y_star_u'].apply(lambda Y_star_u: find_GY_star_lag(Y_star_u, pdata['Score'].tolist(), pdata['Gy'].tolist()))

    #Now, calculate e(y)x
    ey_x = ((((pdata['Px']/100) - pdata['Gy_star_lag'])/(pdata['Gy_star'] - pdata['Gy_star_lag'])) + (pdata['Y_star_u'] - 0.5))
    
    # Return the equated values
    return {'yx':ey_x}


#%%
old = ADM2['x']
new = ADM1['x']

#Testing - works in this instance
equipercen_original(old, new, 0, 50)