
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "North Pacific Meridional Mode index"
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
#=================================================================

def selreg_npmm(var,lon,lat,em=False):
    """
    var: 3D masked array
    lon,lat
    """
    #region
    lon_npmmreg,lat_npmmreg,lev_npmmreg,var_npmmreg = selreg(var, lon, lat, lev=[], \
    lon1=-180%360, lon2=-70%360, lat1=-20, lat2=30, lev1=[], lev2=[], em=em)
    return lon_npmmreg, lat_npmmreg, var_npmmreg

def theilsen_regress_predict(var):
    """
    Input:-
    var: 1-D array var
    regressortype = LinearRegression, TheilSenRegressor

    Output: regression coefficient

    """
    regressor = TheilSenRegressor()
    y = np.asarray(var).reshape(-1,1)
    X = np.arange(len(y)).reshape(-1,1)
    regressor.fit(X,y)
    return regressor.predict(X)

def remove_cti(var,var_npmmreg,lon,lat,em=False):
    """
    sst,sst_npmmreg: 3D or 4D masked array
    lon,lat
    """
    #Calculate the equatorial Pacific Cold tongue Index
    cti = ((selreg(var, lon, lat, lev=[], \
    lon1=-180%360, lon2=-90%360, lat1=-6, lat2=6, lev1=[], lev2=[], em=em)[-1]).mean(axis=-1)).mean(axis=-1)
    #Remove Linear fit of Cold Tongue index
    var_npmmreg_lr = np.ma.subtract(var_npmmreg, \
    theilsen_regress_predict(cti)[...,np.newaxis,np.newaxis])
    return var_npmmreg_lr

def coslat(var_npmmreg, lon_npmmreg, lat_npmmreg,em=False):
    wgtmat = np.cos(np.tile(abs(lat_npmmreg.values[:,None])*np.pi/180,(1,len(lon_npmmreg))))[np.newaxis,...]
    var_npmmreg_c = var_npmmreg*wgtmat
    return var_npmmreg_c

def max_cov_pattern(sst3d_3m, u3d_3m, v3d_3m):
    """
    Input:
    sst3d, u3d, v3d: 3-D masked array (3-month running mean)

    Output:
    V1: SST MCA pattern
    """
    ntime, nrow, ncol = sst3d_3m.shape
    sst2d = np.reshape(sst3d_3m, (ntime, nrow*ncol), order='F')
    sstnonMissingIndex = np.where(np.isnan(sst2d[0]) == False)[0]
    sst2dNoMissing = sst2d[:, sstnonMissingIndex]

    u2d = np.reshape(u3d_3m, (ntime, nrow*ncol), order='F')
    v2d = np.reshape(v3d_3m, (ntime, nrow*ncol), order='F')

    uv2d = np.ma.concatenate((u2d,v2d),axis=-1)
    nonMissingIndex = np.where(np.isnan(uv2d[0]) == False)[0]
    uv2dNoMissing = uv2d[:, nonMissingIndex]

    Cxy = np.dot(uv2dNoMissing.T, sst2dNoMissing)/(ntime-1.0)
    U, s, V = np.linalg.svd(Cxy, full_matrices=False)
    V = V.T

    return V

def exp_coeff(sst3d, sst3d_3m, u3d_3m, v3d_3m, lon, lat, em=False):
    """
    Input:
    sst3d: 3-D array (without 3-month running mean)
    sst3d_3m, u3d_3m, v3d_3m: 3-D array SST, U, V (with 3-month running mean)

    Output:
    EC:
    """
    #Select region
    lon_npmmreg, lat_npmmreg, ssta_npmmreg_3m = selreg_npmm(sst3d_3m,lon,lat,em=em)
    lon_npmmreg, lat_npmmreg, ssta_npmmreg = selreg_npmm(sst3d,lon,lat,em=em)
    lon_npmmreg, lat_npmmreg, u1000a_npmmreg_3m = selreg_npmm(u3d_3m,lon,lat,em=em)
    lon_npmmreg, lat_npmmreg, v1000a_npmmreg_3m = selreg_npmm(v3d_3m,lon,lat,em=em)
    # #Remove CTI
    # ssta_npmmreg_3mlr = remove_cti(sst3d_3m,ssta_npmmreg_3m,lon,lat,em=em)
    # ssta_npmmreg_lr = remove_cti(sst3d,ssta_npmmreg,lon,lat,em=em)
    # u1000a_npmmreg_3mlr = remove_cti(u3d_3m,u1000a_npmmreg_3m,lon,lat,em=em)
    # v1000a_npmmreg_3mlr = remove_cti(v3d_3m,v1000a_npmmreg_3m,lon,lat,em=em)
    #Coslat
    ssta_npmmreg_3mlrc = coslat(ssta_npmmreg_3m, lon_npmmreg, lat_npmmreg,em=em)
    ssta_npmmreg_lrc = coslat(ssta_npmmreg, lon_npmmreg, lat_npmmreg,em=em)
    u1000a_npmmreg_3mlrc = coslat(u1000a_npmmreg_3m, lon_npmmreg, lat_npmmreg,em=em)
    v1000a_npmmreg_3mlrc = coslat(v1000a_npmmreg_3m, lon_npmmreg, lat_npmmreg,em=em)
    #MCA spatial pattern
    V1 = max_cov_pattern(ssta_npmmreg_3mlrc, u1000a_npmmreg_3mlrc, v1000a_npmmreg_3mlrc)
    #Expansion coefficients
    ntime, nrow_sst, ncol_sst = ssta_npmmreg_lrc.shape
    sst2d = np.reshape(ssta_npmmreg_lrc, (ntime, nrow_sst*ncol_sst), order='F')
    sstnonMissingIndex = np.where(np.isnan(sst2d[0]) == False)[0]
    sst2dNoMissing = sst2d[:, sstnonMissingIndex]

    b1 = (np.dot(sst2dNoMissing, V1[:,0, np.newaxis])).squeeze()
    b1_n = ((b1.squeeze() - b1.squeeze().mean())/ b1.squeeze().std())
    return b1, b1_n
