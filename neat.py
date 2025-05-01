# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:00:57 2025

@author: Laura
"""
import pandas as pd
import numpy as np
import itertools
#%%
#Read in data from HW4
formx = pd.read_csv(r'C:\Users\laycocla\OneDrive - James Madison University\Documents\A&M\Equating\HW4\formx.dat',
                    sep='\s+')
formy = pd.read_csv(r'C:\Users\laycocla\OneDrive - James Madison University\Documents\A&M\Equating\HW4\formy.dat',
                    sep='\s+')

#Establish synthetic population items
#X
V_1 = formx['Anchor']
#Y
V_2 = formy['Anchor']
w1 = 0.5
w2 = 0.5

#Calculate gamma
gamma_1 = formx['Anchor'].cov(formx['Uncommon']/formx['Anchor'].var())
gamma_2 = formy['Anchor'].cov(formy['Uncommon']/formy['Anchor'].var())

#Synthetic population stuff
mu_sx = np.mean(formx['Uncommon']) - w2*gamma_1*(np.mean(formx['Anchor']) - np.mean(formy['Anchor']))
mu_sy = np.mean(formy['Uncommon']) + w1*gamma_2*(np.mean(formx['Anchor']) - np.mean(formy['Anchor']))

var_sx = formx['Uncommon'].var() - w2*(gamma_1**2)*(formx['Anchor'].var() - formy['Anchor'].var()) + w1*w2*(gamma_1**2)*(np.mean(formx['Anchor']) - np.mean(formy['Anchor']))**2
var_sy = formy['Uncommon'].var() + w1*(gamma_2**2)*(formx['Anchor'].var() - formy['Anchor'].var()) + w1*w2*(gamma_2**2)*(np.mean(formx['Anchor']) - np.mean(formy['Anchor']))**2

#Get standard deviations
sd_sx = np.sqrt(var_sx)
sd_sy = np.sqrt(var_sy)

scores = list(range(0, 81))

ly_x = (sd_sx/sd_sy)*(scores - mu_sx) + mu_sy

eyx = pd.DataFrame({'Scores': scores,
                   'ey': ly_x})
#%%
from .methods.Tucker import Tucker
from .methods.LevineOS import LevineOS
from .methods.LevineTS import LevineTS
from .methods.FE import FE
from .methods.BH import BH

def neat(x, y, common_x, common_y, score_min, score_max, w1, items = "internal", method = "Tucker"):
    """
    Dispatches a single NEAT equating method.

    Parameters:
    - x, y: Array of raw scores for Form X and Form Y
    - common_x, common_y: Array of anchor scores
    - score_min, score_max: Score range of Form X to equate
    - w1: Weight for group 1 (0 < w1 < 1)
    - items: "internal" or "external" anchor design
    - method: NEAT equating method (options include "Tucker", "LevineOS" (Levine observed score), 
                                    "LevineTS" (Levine true score), "FE" (frequency estimation), 
                                    and "BH"(Braun-Holland))

    Returns:
    - DataFrame of equated scores
    """
    #TODO: Potential errors
        #For internal, common items score should not be larger than total score!

    #Weight validataion
    if not (0 <= w1 <= 1):
        raise ValueError("w1 must be between 0 and 1")
        
    #Method validatation
    valid_methods = ["Tucker", "LevineOS", "LevineTS", "FE", "BH"]
    if method not in valid_methods:
       raise ValueError(f"Method '{method}' not supported. Choose from {valid_methods}")
        
    #Define scores
    scores = np.arange(score_min, score_max + 1)

    if method == "Tucker":
        return Tucker(x, y, common_x, common_y, scores, w1)

    elif method == "LevineOS":
        return allthethings

    elif method == "LevineTS":
        return allthethings

    elif method == "FE":
        return allthethings

    elif method == "BH":
        return allthethings

    else:
        raise ValueError(f"Unsupported method: {method}")
    
    
#%%
    
    #Do we need to add an argument for internal/external common items?
neat(formx['Uncommon'], formy['Uncommon'], formx['Anchor'], formy['Anchor'], 0, 80, w1 = 0.5, method="Tucker")    
 
    
#Read in data
formx <- read.table("formx.dat", header = TRUE)
formy <- read.table("formy.dat", header = TRUE)

#Score vectors
scores_A <- 0:40
scores_U <- 0:80

formx1 <- cbind(formx$Uncommon, formx$Anchor)
formy1 <- cbind(formy$Uncommon, formy$Anchor)

#Tucker method
fx <- freqtab(formx1, scales = list(0:80, 0:40))
fy <- freqtab(formy1, scales = list(0:80, 0:40))

var_2(X) = var_1(X) - alpha^2_1(X|V)[var_1(V)-var_2(V)]
var_1(Y) = var_2(Y) + alpha^2_2(Y|V)[var_1(V)-var_2(V)]

#synthetic pop stuff
mu_s(X) = mu_1(X) - w_2gamma_1[mu_1(V) - mu_2(V)]
mu_s(Y) = mu_2(Y) + w_1gamma_2[mu_1(V) - mu_2(V)]

var_s(X) = var_1(X) - w_2gamma^2_1[var_1(V) - var_2(V)] + w_1w_2gamma^2_1[mu_1(V) - mu_2(V)]^2
var_s(Y) = var_1(Y) + w_1gamma^2_2[var_1(V) - var_2(V)] + w_1w_2gamma^2_2[mu_1(V) - mu_2(V)]^2

gamma_1 = cov(X,V)/var_1(V)
gamma_2 = cov(Y,V)/var_2(V)