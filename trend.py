#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Correlated Mann Kendall Test and Sen Slope Estimate"
__reference__="https://doi.org/10.1007/s00382-020-05369-1"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
#General modules
import bottleneck as bn
from functools import partial
import multiprocessing
import numpy as np
import numpy.ma as ma
import pymannkendall as mk
from scipy import stats
from scipy.stats import norm, rankdata
import xarray as xr

from multiprocessing_py import *
#============================================================
def lag1_acf(x, nlags=1):
    """
    Lag 1 autocorrelation

    Parameters
    ----------
    x : 1D numpy.ndarray
    nlags : Number of lag

    Returns
    -------
    acf : Lag-1 autocorrelation coefficient
    """
    y = x - np.nanmean(x)
    n = len(x)
    d = n * np.ones(2 * n - 1)

    acov = (np.correlate(y, y, 'full') / d)[n - 1:]
    acf = acov[:nlags]/acov[0]
    return acf

def mk_test(x, a=0.10):
    """
    Mann-Kendall test for trend

    Parameters
    ----------
    x : 1D numpy.ndarray
    a : p-value threshold

    Returns
    -------
    trend : tells the trend (increasing, decreasing or no trend)
    h : True (if trend is present or Z-score statistic is greater than p-value) or False (if trend is absent)
    p : p-value of the significance test
    z : normalized test statistics
    Tau : Kendall Tau (s/D)
    s : Mann-Kendal's score
    var_s : Variance of s
    slope : Sen's slope
    """
    #Calculate lag1 acf
    acf = lag1_acf(x)

    r1 = (-1 + 1.96*np.sqrt(len(x)-2))/len(x)-1
    r2 = (-1 - 1.96*np.sqrt(len(x)-2))/len(x)-1
    if (acf > 0) and (acf > r1):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope = mk.yue_wang_modification_test(x)
    elif (acf < 0) and (acf < r2):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope = mk.yue_wang_modification_test(x)
    elif acf == 0:
        #Apply original MK test
        trend, h, p, z, Tau, s, var_s, slope = mk.original_mk_test(x)
    else:
        trend, h, p, z, Tau, s, var_s, slope = np.repeat(np.nan,8)
    return h, p, z, Tau, s, var_s, slope

def new_S_value(S,z,p):
    if abs(z) > p:
        S_n = S
    else:
        S_n = 0.
    return S_n

# standardized test statistic Z
def calc_z_score(s, var_s,a=0.10):
    if (s > 0) and (np.isnan(var_s)==False):
        z = (s - 1)/np.sqrt(var_s)
    elif (s == 0) and (np.isnan(var_s)==False):
        z = 0
    elif (s < 0) and (np.isnan(var_s)==False):
        z = (s + 1)/np.sqrt(var_s)
    else:
        z = np.nan

    p = 2*(1-norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1-a/2)
    return z,p,h

def mask_h(h_matrix,z_matrix,q):
    h_ma = ma.masked_where(h_matrix<1.,h_matrix)
    h_n=ma.masked_where(np.abs(z_matrix)<q,h_ma)
    return h_n



def ensmean_significance_2(trend_data_all,seasons=True):
    if seasons == True:
        #Extract S
        S = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                S.append(trend_data_all[i][j][4])
        #Extract z
        z = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                z.append(trend_data_all[i][j][2])
        #Extract p
        p = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                p.append(trend_data_all[i][j][1])
        #Extract var_s
        var_s = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                var_s.append(trend_data_all[i][j][5])

        #Modify S where trend is insignificant
        v_new_S_value = np.vectorize(new_S_value)
        S_n = []
        for i in range(len(S)):
            S_n.append(v_new_S_value(S[i],z[i],p[i]))

        #Split according to seasons
        S_n_split = [[] for _ in range(4)]
        for index, item in enumerate(S_n):
            S_n_split[index % 4].append(item)

        #Calculate mean variance of S
        var_s_split = [[] for _ in range(4)]
        for index, item in enumerate(var_s):
            var_s_split[index % 4].append(item)

    else:
        #Extract S
        S = [trend_data_all[i][4] for i in range(len(trend_data_all))]
        #Extract z
        z = [trend_data_all[i][2] for i in range(len(trend_data_all))]
        #Extract p
        p = [trend_data_all[i][1] for i in range(len(trend_data_all))]
        #Extract var_s
        var_s = [trend_data_all[i][5] for i in range(len(trend_data_all))]

        #Modify S where trend is insignificant
        v_new_S_value = np.vectorize(new_S_value)
        S_n = v_new_S_value(S,z,p)

        S_n_split = [S_n]
        var_s_split = [var_s]

    #average ensemble members
    S_n_mean = []
    for i in range(len(S_n_split)):
        S_n_mean.append((S_n_split[i][0]+S_n_split[i][1])/2.)

    #average ensemble members
    var_s_mean = []
    for i in range(len(var_s_split)):
        var_s_mean.append((var_s_split[i][0]+var_s_split[i][1])/2.)

    #Calculate z score, p value and h value
    v_z_score = np.vectorize(calc_z_score)
    z_s = []
    p_s = []
    h_s = []
    for i in range(len(S_n_mean)):
        z_s.append(v_z_score(S_n_mean[i],var_s_mean[i])[0])
        p_s.append(v_z_score(S_n_mean[i],var_s_mean[i])[1])
        h_s.append(v_z_score(S_n_mean[i],var_s_mean[i])[2])

    #calculate pfdr
    q = [calc_pfdr(v) for v in p_s]

    #Mask h
    h_n = []
    for i in range(len(S_n_mean)):
        h_n.append(mask_h(h_s[i],z_s[i],q[i]))

    return h_n

def calc_linregress(series,var):
    slope = np.zeros(var.mean(axis=0).shape)
    intercept = np.zeros(var.mean(axis=0).shape)
    r_value = np.zeros(var.mean(axis=0).shape)
    p_value = np.zeros(var.mean(axis=0).shape)
    std_err = np.zeros(var.mean(axis=0).shape)
    for i in range(var.shape[1]):
        for j in range(var.shape[2]):
            x = series
            y = var[:,i,j]
            #if(not np.ma.is_masked(y)):
            slope[i,j], intercept[i,j], r_value[i,j], p_value[i,j], std_err[i,j] = \
            stats.linregress(x,y)
    return slope,intercept,r_value,p_value,std_err

def ensmean_significance(trend_data_all,seasons=True, n=4):
    if seasons == True:
        #Extract S
        S = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                S.append(trend_data_all[i][j][4])
        #Extract z
        z = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                z.append(trend_data_all[i][j][2])
        #Extract p
        p = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                p.append(trend_data_all[i][j][1])
        #Extract var_s
        var_s = []
        for i in range(len(trend_data_all)):
            for j in range(len(trend_data_all[i])):
                var_s.append(trend_data_all[i][j][5])

        #Modify S where trend is insignificant
        v_new_S_value = np.vectorize(new_S_value)
        S_n = []
        for i in range(len(S)):
            S_n.append(v_new_S_value(S[i],z[i],p[i]))

        #Split according to seasons
        S_n_split = [[] for _ in range(4)]
        for index, item in enumerate(S_n):
            S_n_split[index % 4].append(item)

        #Calculate mean variance of S
        var_s_split = [[] for _ in range(4)]
        for index, item in enumerate(var_s):
            var_s_split[index % 4].append(item)

    else:
        #Extract S
        S = [trend_data_all[i][4] for i in range(len(trend_data_all))]
        #Extract z
        z = [trend_data_all[i][2] for i in range(len(trend_data_all))]
        #Extract p
        p = [trend_data_all[i][1] for i in range(len(trend_data_all))]
        #Extract var_s
        var_s = [trend_data_all[i][5] for i in range(len(trend_data_all))]
        #Modify S where trend is insignificant
        v_new_S_value = np.vectorize(new_S_value)
        S_n = v_new_S_value(S,z,p)

        S_n_split = [S_n]
        var_s_split = [var_s]
    if n==5:
        #average ensemble members
        S_n_mean = []
        for i in range(len(S_n_split)):
                S_n_mean.append((S_n_split[i][0]+S_n_split[i][1]+S_n_split[i][2]+S_n_split[i][3]+ S_n_split[i][4])/5.)

        #average ensemble members
        var_s_mean = []
        for i in range(len(var_s_split)):
                var_s_mean.append((var_s_split[i][0]+var_s_split[i][1]+var_s_split[i][2]+var_s_split[i][3]+ var_s_split[i][4])/5.)
    elif n==6:
        #average ensemble members
        S_n_mean = []
        for i in range(len(S_n_split)):
                S_n_mean.append((S_n_split[i][0]+S_n_split[i][1]+S_n_split[i][2]+S_n_split[i][3]+S_n_split[i][4]+S_n_split[i][5])/6.)

        #average ensemble members
        var_s_mean = []
        for i in range(len(var_s_split)):
                var_s_mean.append((var_s_split[i][0]+var_s_split[i][1]+var_s_split[i][2]+var_s_split[i][3]+var_s_split[i][4]+var_s_split[i][5])/6.)

    elif n==2:
        S_n_mean = []
        for i in range(len(S_n_split)):
            S_n_mean.append((S_n_split[i][0]+S_n_split[i][1])/2.)

        var_s_mean = []
        for i in range(len(var_s_split)):
            var_s_mean.append((var_s_split[i][0]+var_s_split[i][1])/2.)

    else:
        S_n_mean = []
        for i in range(len(S_n_split)):
            S_n_mean.append(bn.nansum(S_n_split[i])/n)

        var_s_mean = []
        for i in range(len(var_s_split)):
            var_s_mean.append(bn.nansum(var_s_split[i])/n)
    #Calculate z score, p value and h value
    v_z_score = np.vectorize(calc_z_score)
    z_s = []
    p_s = []
    h_s = []
    for i in range(len(S_n_mean)):
        z_s.append(v_z_score(S_n_mean[i],var_s_mean[i])[0])
        p_s.append(v_z_score(S_n_mean[i],var_s_mean[i])[1])
        h_s.append(v_z_score(S_n_mean[i],var_s_mean[i])[2])

    #calculate pfdr
    q = [calc_pfdr(v) for v in p_s]

    #Mask h
    h_n = []
    for i in range(len(S_n_mean)):
        h_n.append(mask_h(h_s[i],z_s[i],q[i]))

    return h_n
