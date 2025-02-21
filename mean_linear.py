# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:54:01 2025

@author: Laura
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

#Get data
ADM1 = pd.read_csv('form_y.csv')
ADM2 = pd.read_csv('form_x.csv')


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

#Making sure I did it right
print(scores)

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

#Check work
print(pdata.head(10))

#Calculate f(x) and g(y)
pdata['f_x'] = pdata['X']/sum(pdata['X'])
pdata['g_y'] = pdata['Y']/sum(pdata['Y'])

#Calculate F(x) and G(y)
pdata['Fx'] = np.cumsum(pdata['f_x'])
pdata['Gy'] = np.cumsum(pdata['g_y'])

pdata['Px'] = 100*(pdata['Fx'].shift(1) + (pdata['f_x']/2))
pdata['Qy'] = 100*(pdata['Gy'].shift(1) + (pdata['g_y']/2))

#First moment: mean
pmean1_X = sum(pdata['Score']*pdata['f_x'])
pmean1_Y = sum(pdata['Score']*pdata['g_y'])

#Second moment: variance
pvar1_X = sum(((pdata['Score'] - pmean1_X)**2)*pdata['f_x'])
pvar1_Y = sum(((pdata['Score'] - pmean1_Y)**2)*pdata['g_y'])

#Turn into SD
psd1_X = np.sqrt(pvar1_X)
psd1_Y = np.sqrt(pvar1_Y)

#Third moment: skew
pskew1_X = (sum((((pdata['Score'] - pmean1_X)**3))*pdata['f_x']))/(psd1_X**3)
pskew1_Y = (sum((((pdata['Score'] - pmean1_Y)**3))*pdata['g_y']))/(psd1_Y**3)

#Fourth moment: kurtosis
pskew1_X = (sum((((pdata['Score'] - pmean1_X)**4))*pdata['f_x']))/(psd1_X**4)
pskew1_Y = (sum((((pdata['Score'] - pmean1_Y)**4))*pdata['g_y']))/(psd1_Y**4)

#First, make G(y) * 100 value column - easier to reference this
pdata['Gy_100'] = pdata['Gy']*100

#Make a function to take each P(x) value and find the smallest Gy_100 value that is => to it

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
pdata['ey_x'] = ((((pdata['Px']/100) - pdata['Gy_star_lag'])/(pdata['Gy_star'] - pdata['Gy_star_lag'])) + (pdata['Y_star_u'] - 0.5))

#First moment: mean
pmean2_X = sum(pdata['ey_x']*pdata['f_x'])
pmean2_Y = sum(pdata['Score']*pdata['g_y'])

#Second moment: variance
pvar2_X = sum(((pdata['ey_x'] - pmean2_X)**2)*pdata['f_x'])
pvar2_Y = sum(((pdata['Score'] - pmean2_Y)**2)*pdata['g_y'])

#Turn into SD
psd2_X = np.sqrt(pvar2_X)
psd2_Y = np.sqrt(pvar2_Y)

#Third moment: skew
pskew2_X = (sum((((pdata['ey_x'] - pmean2_X)**3))*pdata['f_x']))/(psd2_X**3)
pskew2_Y = (sum((((pdata['Score'] - pmean2_Y)**3))*pdata['g_y']))/(psd2_Y**3)

#Fourth moment: kurtosis
pskew2_X = (sum((((pdata['ey_x'] - pmean2_X)**4))*pdata['f_x']))/(psd2_X**4)
pskew2_Y = (sum((((pdata['Score'] - pmean2_Y)**4))*pdata['g_y']))/(psd2_Y**4)

pdata['myx'] = pdata['Score'] + pmean1_Y - pmean1_X

#Linear equating:
#Calculate sd proportion first
a = (psd1_Y/psd1_X)
B = (pmean1_Y - a*pmean1_X)

pdata['lyx'] = a*pdata['Score'] + B

#Need the identity line
pdata['ix'] = 0

#Since manually did equating, don't need to add in here.
plt.plot(pdata['Score'], (pdata['myx'] - pdata['Score']), label = "Mean Equating", color = "#440154FF")
plt.scatter(pdata['Score'], (pdata['myx'] - pdata['Score']), color = "#440154FF", marker = 'x')
plt.plot(pdata['Score'], (pdata['lyx'] - pdata['Score']), label = "Linear Equating", color = "#414487FF")
plt.scatter(pdata['Score'], (pdata['lyx'] - pdata['Score']), color = "#414487FF", marker = 'd')
plt.plot(pdata['Score'], (pdata['ey_x'] - pdata['Score']), label = "Equipercentile Equating", color = "#22A884FF")
plt.scatter(pdata['Score'], (pdata['ey_x'] - pdata['Score']), color = "#22A884FF", marker = 'v')
plt.plot(pdata['Score'], pdata['ix'], label = "Identity", color = "#FDE725FF")
plt.scatter(pdata['Score'], pdata['ix'], color = "#FDE725FF", marker = '.')
plt.xlabel("Raw Score")
plt.ylabel("Equated Score - Raw Score")
plt.title("Difference Between Equated and Raw Scores")
plt.legend()

plt.show()

#If need to clear the plot
plt.cla()

#%%
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#Making sure I did it right
print(scores)

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

#Something isn't working with freqtab() - TODO
test = freqtab(pdata)

linear(test['X'], test['Y'])

linear(test, type = "mean")
