#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "ENSO indices"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
#General modules
import dask.array as da
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts')
import argparse
import numpy as np
import numpy.ma as ma
import xarray as xr
from eofs.standard import Eof
import scipy
from sklearn.linear_model import TheilSenRegressor

#My modules
from read import *
from climanom import *
#============================================================

def pdo_index(ssta, lon, lat):
    ssta_np = selreg(ssta, lon, lat, [None], lat1=20, lat2=90 , lon1=140, lon2=250, lev1=[], lev2=[])[-1]
    lats = lat.sel(latitude=slice(19,90))
    lons = lon.sel(longitude=slice(140,250))
    #Calculate EOF1
    eof_ssta_np = calc_eof(ssta_np.values, lats, 1, eoftype='cov')
    pdo = eof_ssta_np[1].squeeze()
    #Normalise the index
    pdo_sd = np.divide((pdo - pdo.mean()), pdo.std())
    #Pass 6-year low pass filter
    nmonths=6*12
    order=4
    fs = 1
    pdo_f = butter_filter(pdo,nmonths,None,None,fs,order,btype='low')
    #Normalise the filtered index
    pdo_f_sd = np.divide((pdo_f - pdo_f.mean()), pdo_f.std())
    return pdo, pdo_sd, pdo_f, pdo_f_sd

def tripole_index(ssta, lon, lat):
    #Select regions
    reg1 = wgtvar(ma.masked_invalid(ssta.sel(latitude=slice(25,35),\
    longitude=slice(140,-145%360))), \
    lon.sel(longitude=slice(140,-145%360)).values,\
    lat.sel(latitude=slice(25,35)).values, lev=[], ensemble=False)
    reg2 = wgtvar(ma.masked_invalid(ssta.sel(latitude=slice(-10,10),\
    longitude=slice(170,-90%360))), \
    lon.sel(longitude=slice(170,-90%360)).values,\
    lat.sel(latitude=slice(-10,10)).values, lev=[], ensemble=False)
    reg3 = wgtvar(ma.masked_invalid(ssta.sel(latitude=slice(-50,-15),\
    longitude=slice(150,-160%360))), \
    lon.sel(longitude=slice(150,-160%360)).values,\
    lat.sel(latitude=slice(-50,-15)).values, lev=[], ensemble=False)
    #Calculate index
    tpi = reg2 - ((reg1+reg3)/2.)
    #Normalise the TPI index
    tpi_sd = (tpi - tpi.mean())/ tpi.std()
    #Pass a 13-year Chebyshev low pass filter
    order=4
    nmonths=6*12
    fs=1
    Wn = (1/nmonths)/(0.5*fs)
    b,a = scipy.signal.cheby2(order,25, Wn, btype='low')
    tpi_f = scipy.signal.filtfilt(b, a, tpi)
    #Normalise filtered TPI index
    tpi_f_sd = (tpi_f - tpi_f.mean())/ tpi_f.std()
    return tpi, tpi_sd, tpi_f, tpi_f_sd
