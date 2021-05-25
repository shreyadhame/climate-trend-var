#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Linear regression, Lagged correlation and regression"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#==============================================================================
import numpy as np
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import scipy.stats as stats
import pandas as pd
#My modules

#==============================================================================
def crosscorr(x, y, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    x, y : numpy array objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    df = pd.DataFrame({'X':x, 'Y':y})
    datax = df.X
    datay = df.Y

    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

def runningcorr(x, y, N):
    """
    Applies running correlation to 1D array using running window
    """
    # Initilize placeholder array
    r = np.zeros((len(x) - (N - 1),))
    p = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):

         r[i] = stats.pearsonr(x[i:(i + N)],y[i:(i + N)])[0]
         p[i] = stats.pearsonr(x[i:(i + N)],y[i:(i + N)])[1]

    return r, p

def theilsen_regress_coeff(var, a):
    """
    Input:-
    var: 1-D array var
    a: 1-D array index
    regressortype = LinearRegression, TheilSenRegressor

    Output: regression coefficient

    """
    regressor = TheilSenRegressor()
    y = np.asarray(var).reshape(-1,1)
    X = a.reshape(-1,1)
    regressor.fit(X,y)
    return np.array([regressor.coef_])

def crossregress(var, a, lag=0):
    """
    Input:-
    var: 1-D array var
    a: 1-D array index

    Output: regression coefficient

    """
    shiftedy = np.roll(a, lag)
    r = stats.theilslopes(var, shiftedy)[0]
    return r

def calc_rs(var, a, lag=0):
    """
    Input:-
    var: 1-D array var
    a: 1-D array index

    Output: regression coefficient

    """
    rs = [crossregress(var,a,lag) for lag in range(-int(lag),int(lag+1))]
    return rs
