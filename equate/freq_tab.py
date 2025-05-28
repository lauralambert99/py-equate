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

fs_x = w1*f1x + w2*f2x
gs_y = w1*g1y + w2*g2y


