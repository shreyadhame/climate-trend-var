#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Power spectral density"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import mlab

from spectrum import pcovar
#============================================================

def pcovar(var):
    """
    Input: SSTA time series
    """

    nw   = 48  # order of an autoregressive prediction model for the signal, used in estimating the PSD.
    nfft = 256 # NFFT (int) – total length of the final data sets (padded with zero if needed

    fs  = 1     # default value
    p   = pcovar(var, nw, nfft, fs)
    return p

def welch(var):
    """
    Input: SSTA time series
    """
    n        = 150
    alpha    = 0.5
    noverlap = 75
    nfft     = 256 #default value
    fs       = 1   #default value
    win      = signal.tukey(n, alpha)
    ssta     = var.reshape(360) # convert vector

    f1, pxx1  = signal.welch(ssta, nfft=nfft, fs=fs, window=win, noverlap=noverlap)
    return f1, pxx1
