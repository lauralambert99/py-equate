# -*- coding: utf-8 -*-
"""
Created on Wed May 14 08:17:41 2025

@author: Laura
"""
import pandas as pd
import numpy as np

#Test data
formx = pd.read_csv("C:\\Users\\laycocla\\OneDrive - James Madison University\\Documents\\A&M\\Equating\\HW4\\formx.dat",
                    sep='\\s+')
formy = pd.read_csv("C:\\Users\\laycocla\\OneDrive - James Madison University\\Documents\\A&M\\Equating\\HW4\\formy.dat",
                    sep='\\s+')
w1 = 0.5
w2 = 0.5

scores = np.arange(0, 80 + 1)

rangex = np.arange(0, 81)
rangev = np.arange(0, 41)

table_x = joint_distribution(formx, 'Uncommon', 'Anchor', rangex, rangev)
table_y = joint_distribution(formy, 'Uncommon', 'Anchor', rangex, rangev)

table_x_v2 = common_item_marginal(table_x)
table_y_v2 = common_item_marginal(table_y)

cond_x = conditional_distribution(table_x_v2)
cond_y = conditional_distribution(table_y_v2)

cond_x_pop2 = reweight_conditional_distribution(cond_x, other_marginals = cond_y.iloc[-1])
cond_y_pop1 = reweight_conditional_distribution(cond_y, other_marginals = cond_x.iloc[-1])

f1x = table_x['Marginal']
f2x = cond_x_pop2.iloc[:-1]['Marginal']

g1y = cond_y_pop1.iloc[:-1]['Marginal']
g2y = table_y['Marginal']

fsx = w1*f1x + w2*f2x
gsy = w1*g1y + w2*g2y

Fsx = fsx.cumsum()
Gsy = gsy.cumsum()

Psx = 100*(Fsx.shift(1, fill_value = 0) + fsx/2)

#Make G(y) * 100 value column - easier to reference this
Gsy_100 = Gsy*100

#Make a function to take each P(x) value and find the smallest Gy_100 value that is => to it

#But what we really want is the corresponding Y value

def find_Y_star(Psx, Gsy, Y):
    idx = np.searchsorted(Gsy, Psx, side="left")
    return Y[idx] if idx < len(Y) else None

scores = scores
  
yvals = formy['Uncommon'].value_counts().reindex(scores, fill_value=0).sort_index()
  
pdata = pd.DataFrame({
    'Score': scores, 
    'Y': yvals.values, 
    'fsx': fsx.values, 
    'gsy': gsy.values, 
    'Fsx': Fsx.values, 
    'Gsy': Gsy.values,
    'Psx': Psx.values,
    'Gsy_100': Gsy_100.values
})
#%%
import pandas as pd
import numpy as np

#This gets the joint distributions of uncommon and common items
#Also calculates marginal and cumulative distributions for the uncommon items
#Need to add marginal distibution for common items as a row
def joint_distribution(df, form_col, common_col, form_range = None, common_range = None):
    """
    Create a joint probability distribution of form score vs common-item score, 
    including marginal and cumulative probabilities.
    
    Parameters:
        df (pd.DataFrame): DataFrame with score columns
        form_col (str): Column name for form score
        common_col (str): Column name for common-item score
        form_range (list or None): Optional list of form score values to include
        common_range (list or None): Optional list of common score values to include
        
    Returns:
        pd.DataFrame: Joint distribution with marginal and cumulative probabilities
    """
    
    #Determine score ranges from data if not provided
    if form_range is None:
        form_range = range(df[form_col].min(), df[form_col].max() + 1)

    if common_range is None:
        common_range = range(df[common_col].min(), df[common_col].max() + 1)


    #Make an empty joint frequency table
    joint = pd.DataFrame(0, index = form_range, columns = common_range)
    
    #Count joint occurrences
    for _, row in df.iterrows():
        x = row[form_col]
        v = row[common_col]
        if x in joint.index and v in joint.columns:
            joint.at[x, v] += 1
        else:
            #Optionally add missing score to table instead of skipping
            joint = joint.reindex(index=sorted(set(joint.index).union({x})))
            joint = joint.reindex(columns=sorted(set(joint.columns).union({v})))
            joint.at[x, v] = 1  #Initialize to 1 (was 0, now first seen)

    #Normalize to probabilities
    total = joint.to_numpy().sum()
    joint_prob = joint / total

    #Marginal and cumulative distributions
    joint_prob['Marginal'] = joint_prob.sum(axis = 1)
    joint_prob['Cumulative'] = joint_prob['Marginal'].cumsum()

    return joint_prob

#%%
import pandas as pd

def common_item_marginal(joint_probs: pd.DataFrame, exclude_last_n: int = 2) -> pd.DataFrame:
    """
    Adds a new row with the marginal distribution of common-item scores
    by summing over rows, excluding the last `exclude_last_n` columns.
    
    Using this function immediately following the joint_distribution function
    creates an initial common-item and Form distribution for a population
    
    Parameters:
    - joint_probs: DataFrame containing the joint probabilities.
    Feed the joint_distribution DataFrame straight to this
    
    - exclude_last_n: How many columns to exclude from the marginal row. (2)
    
    Returns:
    - Modified DataFrame with a new marginal row added.
    """
    # Identify which columns to include in the marginal
    include_cols = joint_probs.columns[:-exclude_last_n] if exclude_last_n > 0 else joint_probs.columns
    marginal_row = joint_probs[include_cols].sum(axis = 0)

    # Fill the excluded columns with NaNs to maintain shape
    for col in joint_probs.columns[-exclude_last_n:]:
        marginal_row[col] = pd.NA

    # Append the marginal row with a label
    marginal_row.name = "Marginal_Common"
    return pd.concat([joint_probs, pd.DataFrame([marginal_row])])


#%%
import pandas as pd

def conditional_distribution(joint_probs: pd.DataFrame, marginal_row_label="Marginal_Common", exclude_last_n=2) -> pd.DataFrame:
    """
    Computes the conditional distribution of total scores given common-item scores
    by dividing each column by the marginal distribution of the common-item scores.

    Parameters:
    - joint_probs: DataFrame with joint probabilities including a row labeled `marginal_row_label`.
    - marginal_row_label: Name of the row containing marginal distribution for the common items.
    If feeding straigh in, will be "Marginal_Common"
    - exclude_last_n: Number of columns at the end to exclude from the conditional computation.

    Returns:
    - DataFrame of the same shape, with conditional probabilities replacing the joint ones,
      and NaNs in excluded columns and the marginal row.
    """
    cond_probs = joint_probs.copy()

    #Columns to divide over
    process_cols = cond_probs.columns[:-exclude_last_n] if exclude_last_n > 0 else cond_probs.columns

    #Extract marginal row values for divisor
    marginals = cond_probs.loc[marginal_row_label, process_cols]

    #Apply division to all rows EXCEPT the marginal row - don't want it messed up
    for row in cond_probs.index:
        if row != marginal_row_label:
            for col in process_cols:
                denom = marginals[col]
                cond_probs.at[row, col] = cond_probs.at[row, col] / denom if denom != 0 else pd.NA

    return cond_probs

#%%
import pandas as pd

def reweight_conditional_distribution(
    conditional_probs: pd.DataFrame,
    other_marginals: pd.Series,
    marginal_row_label = "Marginal_Common",
    exclude_last_n = 2
) -> pd.DataFrame:
    """
    Computes expected score distribution by reweighting conditional probabilities 
    using marginal distribution from the other population.

    Parameters:
    - conditional_probs: DataFrame with conditional probabilities P(score | common).
    - other_marginals: Series with marginal distribution of common scores from other group.
    - marginal_row_label: Row label to skip in conditional_probs.
    - exclude_last_n: Number of trailing columns to exclude (e.g., totals/cumulatives).

    Returns:
    - DataFrame with reweighted expected distribution.
    """
    weighted_df = conditional_probs.copy()

    #Columns over which to apply weights (exclude totals)
    process_cols = weighted_df.columns[:-exclude_last_n] if exclude_last_n > 0 else weighted_df.columns

    for row in weighted_df.index:
        if row != marginal_row_label:
            for col in process_cols:
                marginal_value = other_marginals.get(col, pd.NA)
                if pd.notna(marginal_value):
                    weighted_df.at[row, col] = weighted_df.at[row, col] * marginal_value
                else:
                    weighted_df.at[row, col] = pd.NA

    return weighted_df

#%%
#bh testing
tablex = joint_distribution(formx, 'Uncommon', 'Anchor', rangex, rangev)
tabley = joint_distribution(formy, 'Uncommon', 'Anchor', rangex, rangev)


g1x_v2 = common_item_marginal(tablex)

g2y_v2 = common_item_marginal(tabley)


#Then, make conditional distribution tables
cond_x = conditional_distribution(g1x_v2)
cond_y = conditional_distribution(g2y_v2)

#Calculate the opposite distributions for the forms
#i.e., distribution of Form Y in population 1
cond_x_pop2 = reweight_conditional_distribution(cond_x, other_marginals = cond_y.iloc[-1])
cond_y_pop1 = reweight_conditional_distribution(cond_y, other_marginals = cond_x.iloc[-1])

#Calculate synthetic population values
f1x = tablex['Marginal']
f2x = cond_x_pop2.iloc[:-1]['Marginal']

g1y = cond_y_pop1.iloc[:-1]['Marginal']
g2y = tabley['Marginal']


#Marginal synthetic distributions
fsx = w1*f1x + w2*f2x
gsy = w1*g1y + w2*g2y

x = formx['Uncommon'].value_counts().reindex(scores, fill_value=0).sort_index()
y = formy['Uncommon'].value_counts().reindex(scores, fill_value=0).sort_index()  

#Calculate means and standard deviations
mu_sx = sum(x*fsx)  #Mean of synthetic population
var_sx = sum(((x-mu_sx)**2)*fsx)  #Variance of synthtic population
sd_sx = var_sx**0.5
  
mu_sy = sum(y*gsy)  #Mean of synthetic population
var_sy = sum(((y-mu_sy)**2)*gsy)  #Variance of synthtic population
sd_sy = var_sy**0.5
  
slope = sd_sy / sd_sx  #Ratio of standard deviations
 
intercept = mu_sy - slope * mu_sx  
  
ex = slope * scores + intercept
  
eq =  pd.DataFrame({'Score': scores,
                  'ex': ex})
