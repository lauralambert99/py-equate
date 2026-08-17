# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:30:15 2025

@author: Laura
"""

#TODO: question - do we want full possible range of scores, or just reported scores?

import numpy as np
import pandas as pd

def chained(x, y, common_x, common_y, score_min, score_max, anchor_min, anchor_max, type = "linear"):
    """
  Perform chained equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  
  score_min: minimum possible score
  score_max: maximum possible score
  
  anchor_min: minimum possible anchor score
  anchor_max: maximum possible anchor score
  
  type: of chained equating; options are "linear" and "eq" (equipercentile).  Default is "linear".
  

  Returns:
  DataFrame of equated scores
  """
    #Define scores
    scores = np.arange(score_min, score_max + 1)
    
    if type == "linear":  
         
      #Calculate gammas
      gamma_1 = np.std(x, ddof = 1)/np.std(common_x, ddof = 1)
      gamma_2 = np.std(y, ddof = 1)/np.std(common_y, ddof = 1)

      #Calculate Means
      mean_x = np.mean(x)
      mean_y = np.mean(y)
    
      mean_cx = np.mean(common_x)
      mean_cy = np.mean(common_y)
    
    
      #Chained equating equation
      lyx = (mean_y + gamma_2*(mean_cx - mean_cy) - (gamma_2/gamma_1)*mean_x) + (gamma_2/gamma_1)*scores
    
      eyx = pd.DataFrame({'Scores': scores,
                       'ey': lyx})
      return eyx
    
    elif type == "eq":
        #Form X to common in Population 1
        pfreq_x = x.value_counts().reindex(range(score_min, score_max + 1), fill_value=0).sort_index()
        pfreq_v = common_x.value_counts().reindex(range(anchor_min, anchor_max + 1), fill_value=0).sort_index()
       
        #Calculate f(x) and h1(v)
        f_x = pfreq_x/pfreq_x.sum()
        h_v = pfreq_v/pfreq_v.sum()
        
        #Calculate F(x) and H1(v)
        Fx = f_x.cumsum()
        Hv = h_v.cumsum()
        
        Px = 100*(Fx.shift(1, fill_value = 0) + f_x/2)
        
        #Make H(v) * 100 value column - easier to reference this
        Hv_100 = Hv*100

        #Make a function to take each P(x) value and find the smallest Hv_100 value that is => to it
        def find_V_star(Px, Hv, V):
          idx = np.searchsorted(Hv, Px, side="left")
          return V[idx] if idx < len(V) else None

        pop1 = pd.DataFrame({
            'Score': scores,
            'Anchor_scores': np.arange(anchor_min, anchor_max + 1), 
            'X': pfreq_x.values, 
            'V': pfreq_v.values, 
            'f_x': f_x.values, 
            'h_v': h_v.values, 
            'Fx': Fx.values, 
            'Hv': Hv.values,
            'Px': Px.values,
            'Hv_100': Hv_100.values
        })

        #Use the function to make a new column 
        pop1['V_star_u'] = pop1['Px'].apply(lambda Px: find_V_star(Px, pop1['Hv_100'], pop1['Anchor_scores']))

        #Compute G(Y*u)
        pop1['Hv_star'] = pop1['V_star_u'].apply(lambda v_star: Hv[v_star] if pd.notna(v_star) else None)
        
        #Will also need a lag Y*u for equation
        pop1['Hv_star_lag'] = pop1['V_star_u'].apply(lambda v_star: Hv[v_star - 1] if pd.notna(v_star) and v_star > score_min else None)

        #Compute equated scores
        pop1['e_vx'] = ((((pop1['Px'] / 100) - pop1['Hv_star_lag']) / 
                          (pop1['Hv_star'] - pop1['Hv_star_lag'])) + 
                          (pop1['V_star_u'] - 0.5))

        #Common to Form Y in Population 2
        pfreq_v2 = common_y.value_counts().reindex(range(anchor_min, anchor_max + 1), fill_value=0).sort_index()
        pfreq_y = y.value_counts().reindex(range(score_min, score_max + 1), fill_value=0).sort_index()

        h2_v = pfreq_v2 / pfreq_v2.sum()
        g_y = pfreq_y / pfreq_y.sum()

        H2 = h2_v.cumsum()
        Gy = g_y.cumsum()

        Pv = 100 * (H2.shift(1, fill_value=0) + h2_v / 2)
        Qy = 100 * (Gy.shift(1, fill_value=0) + h2_v / 2)

        #Make G(y) * 100 column for lookup
        Gy_100 = Gy * 100

        def find_Y_star(Qy, Gy, Y):
            idx = np.searchsorted(Gy, Qy, side='left')
            return Y[idx] if idx < len(Y) else None

        pop2 = pd.DataFrame({
            'Score': scores,
            'V': np.arange(anchor_min, anchor_max + 1),
            'Pv': Pv.values,
            'Qy': Qy.values,
            'h2_v': h2_v.values,
            'H2': H2.values,
            'Gy_100': Gy_100.values,
            'G_y': Gy.values
            })

        #Make Ystar column
        pop2['Y_star_u'] = pop2['Qy'].apply(lambda Qy: find_Y_star(Qy, Gy_100.values, scores))
        
        #Compute G(Y*u)
        pop2['Gy_star'] = pop2['Y_star_u'].apply(lambda y_star: Gy[y_star] if pd.notna(y_star) else None)
        
        #Compute lag Y*u
        pop2['Gy_star_lag'] = pop2['Y_star_u'].apply(lambda y_star: Gy[y_star - 1] if pd.notna(y_star) and y_star > score_min else None)

        #Compute equated scores
        pop2['e_yv'] = (((pop2['Pv'] / 100) - pop2['Gy_star_lag']) /
                  (pop2['Gy_star'] - pop2['Gy_star_lag']) + 
                  (pop2['Y_star_u'] - 0.5))
        
        #Put the above together
        #Struggled here - no clear formula
        #Went with interpolation for now??? (i.e., look here, get this, use that there to get the other thing)
        
        #Interpolation requires sorted values 
        interp_input = pop2.sort_values('V')

        #Map e_vx to e_yv
        pop1['e_yx'] = np.interp(
            pop1['e_vx'],        #x-values to map
            interp_input['V'],         #known x: V
            interp_input['e_yv']       #known y: equated Y scores
            )
        
        return pd.DataFrame({'Score': scores,
                            'e_yx': pop1['e_yx']})
            
        
    else:
        raise ValueError(f"Unsupported type: {type}")