#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Calculate significance"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#==============================================================================
import numpy as np
import scipy
from scipy.stats import t
from multiprocessing import *

#==============================================================================
def lag_acf(x, nlags=1):
    """
    Lag autocorrelation

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

def calc_correlated_t(a, b):
    """
    Correlated Student's t-Test

    Parameters
    ----------
    a : 1-D numpy.ndarray
    b : 1-D numpy.ndarray

    Returns
    -------
    p : p-value
    """
    #Calculate t statistic
    tstat = scipy.stats.ttest_ind(a, b, axis=0, equal_var=False)[0]
    #Effective degrees of freedom
    N = len(a)
    neff_inv = (1/N)+(2/N)*(sum(map(lambda j: ((N-j)/N)*lag_acf(a,nlags=j)*lag_acf(b,nlags=j), \
    range(N))))
    df = 1/neff_inv
    # #Lag-1 autocorrelation
    # acfa = lag1_acf(a)
    # acfb = lag1_acf(b)
    # #Degrees of freedom
    # na = len(a)*((1-acfa)/(1+acfa))
    # nb = len(b)*((1-acfb)/(1+acfb))
    # df = na + nb - 2
    #p-value
    p = (1.0 - t.cdf(abs(tstat), df)) * 2.0
    return p

def calc_pfdr(p_matrix,a=0.10):
    """
    Calculate p_fdr
    """
    p_sort = np.sort(p_matrix.flatten()[~np.isnan(p_matrix.flatten())])

    q = []
    for i in range(len(p_sort)):
        if p_sort[i] <= (i/len(p_sort))*a:
            q.append(p_sort[i])
        else:
            pass
    if q==[]:
        return a
    else:
        return np.max(q)
