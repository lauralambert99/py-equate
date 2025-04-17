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
formx = pd.read_csv(r'C:\Users\Laura\OneDrive - James Madison University\Documents\A&M\Equating\HW4\formx.dat',
                    sep='\s+')
formy = pd.read_csv(r'C:\Users\Laura\OneDrive - James Madison University\Documents\A&M\Equating\HW4\formy.dat',
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

def neat(x, y, common_x, common_y, score_min, score_max, w1, items = "internal", method = "Tucker"):
    """
    A function to perform single group equating.

    Parameters:
    x : array of scores on Form X
    y : array of scores on Form Y 
    common: array of item numbers that are common between Form X and Form Y
    score_min: minimum score on the form
    score_max: maximum score on the form
    w1: weight for group 1
    items: str, optional
        If anchor items are internal ("internal") or external ("external").
        Default is "internal".
    method: str, optional
        Type of NEAT equating method.
        Options include linear methods (Tucker as "Tucker", Levine Observed score as "LevineOS", and Levine True Score as "LevineTS"), equipercentile ("FE"), Braun-Holland ("BH").
        Default is Tucker.
    


    Returns:
    DataFrame
        A DataFrame containing yx values for equated x scores
    """
    #TODO: Potential errors
        #For internal, common items score should not be larger than total score!
        #Weights can't be negative!!
        #Method needs to be one of the ones listed
     
    #Define scores
    scores = np.arange(score_min, score_max + 1)
    
    #Define weights
    w1 = w1
    w2 = (1 - w1)
    
    #Calculate gamma
    gamma_1 = common_x.cov(x/common_x.var())
    gamma_2 = common_y.cov(y/common_y.var())

    #Synthetic population stuff
    mu_sx = np.mean(x) - w2*gamma_1*(np.mean(common_x) - np.mean(common_y))
    mu_sy = np.mean(y) + w1*gamma_2*(np.mean(common_x) - np.mean(common_y))

    var_sx = x.var() - w2*(gamma_1**2)*(common_x.var() - common_y.var()) + w1*w2*(gamma_1**2)*(np.mean(common_x) - np.mean(common_y))**2
    var_sy = y.var() + w1*(gamma_2**2)*(common_x.var() - common_y.var()) + w1*w2*(gamma_2**2)*(np.mean(common_x) - np.mean(common_y))**2

    #Get standard deviations
    sd_sx = np.sqrt(var_sx)
    sd_sy = np.sqrt(var_sy)
    
    ly_x = (sd_sx/sd_sy)*(scores - mu_sx) + mu_sy

    eyx = pd.DataFrame({'Scores': scores,
                       'ey': ly_x})
    return eyx

#%%
    
    #Do we need to add an argument for internal/external common items?
neat(formx['Uncommon'], formy['Uncommon'], formx['Anchor'], formy['Anchor'], 0, 80, 0.5)    
 
    
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