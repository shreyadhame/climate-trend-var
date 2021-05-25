
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Atlantic Meridional Mode index: Average and Maximum Covariance"
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
import klepto
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

def selreg_amm(var,lon,lat,em=False):
    """
    var: 3D masked array
    lon,lat
    """
    #region 1
    lon1,lat1,lev1,var_ammreg1 = selreg(var, lon, lat, lev=[], \
    lon1=-74%360, lon2=360, lat1=-20, lat2=30, lev1=-999., lev2=-999., em=em)
    #region 2
    lon2,lat2,lev2,var_ammreg2 = selreg(var, lon, lat, lev=[], \
    lon1=0, lon2=15, lat1=-20, lat2=30, lev1=-999., lev2=-999., em=em)
    #concatenate along lon axis
    var_ammreg = np.ma.concatenate((var_ammreg1, var_ammreg2), axis=-1)
    lon_ammreg = np.ma.concatenate((lon1,lon2))
    lat_ammreg = lat1
    return lon_ammreg, lat_ammreg, var_ammreg

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

def remove_cti(var,var_ammreg,lon,lat,em=False):
    """
    sst,sst_ammreg: 3D or 4D masked array
    lon,lat
    """
    #Calculate the equatorial Pacific Cold tongue Index
    cti = ((selreg(var, lon, lat, lev=[], \
    lon1=-180%360, lon2=-90%360, lat1=-6, lat2=6, lev1=-999., lev2=-999., em=em)[-1]).mean(axis=-1)).mean(axis=-1)
    #Remove Linear fit of Cold Tongue index
    var_ammreg_lr = np.ma.subtract(var_ammreg, \
    theilsen_regress_predict(cti)[...,np.newaxis,np.newaxis])
    return var_ammreg_lr

def coslat(var_ammreg, lon_ammreg, lat_ammreg,em=False):
    wgtmat = np.cos(np.tile(abs(lat_ammreg.values[:,None])*np.pi/180,(1,len(lon_ammreg))))[np.newaxis,...]
    var_ammreg_c = var_ammreg*wgtmat
    return var_ammreg_c

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

    uv2d = np.stack([np.ma.append(u2d[i],v2d[i],axis=-1) for i in range(len(u2d))])
    nonMissingIndex = np.where(np.isnan(uv2d[0]) == False)[0]
    uv2dNoMissing = uv2d[:, nonMissingIndex]

    Cxy = np.dot(sst2dNoMissing.T, uv2dNoMissing)/(ntime-1.0)
    U, s, V = np.linalg.svd(Cxy, full_matrices=False)
    V = V.T

    return U

def exp_coeff(sst3d, sst3d_3m, u3d_3m, v3d_3m, lon, lat, em=False):
    """
    Input:
    sst3d: 3-D array (without 3-month running mean)
    sst3d_3m, u3d_3m, v3d_3m: 3-D array SST, U, V (with 3-month running mean)

    Output:
    EC:
    """
    #Select region
    lon_ammreg, lat_ammreg, ssta_ammreg_3m = selreg_amm(sst3d_3m,lon,lat,em=em)
    lon_ammreg, lat_ammreg, ssta_ammreg = selreg_amm(sst3d,lon,lat,em=em)
    lon_ammreg, lat_ammreg, u1000a_ammreg_3m = selreg_amm(u3d_3m,lon,lat,em=em)
    lon_ammreg, lat_ammreg, v1000a_ammreg_3m = selreg_amm(v3d_3m,lon,lat,em=em)
    # #Remove CTI
    # ssta_ammreg_3mlr = remove_cti(sst3d_3m,ssta_ammreg_3m,lon,lat,em=em)
    # ssta_ammreg_lr = remove_cti(sst3d,ssta_ammreg,lon,lat,em=em)
    # u1000a_ammreg_3mlr = remove_cti(u3d_3m,u1000a_ammreg_3m,lon,lat,em=em)
    # v1000a_ammreg_3mlr = remove_cti(v3d_3m,v1000a_ammreg_3m,lon,lat,em=em)
    #Coslat
    ssta_ammreg_3mlrc = coslat(ssta_ammreg_3m, lon_ammreg, lat_ammreg,em=em)
    ssta_ammreg_lrc = coslat(ssta_ammreg, lon_ammreg, lat_ammreg,em=em)
    u1000a_ammreg_3mlrc = coslat(u1000a_ammreg_3m, lon_ammreg, lat_ammreg,em=em)
    v1000a_ammreg_3mlrc = coslat(v1000a_ammreg_3m, lon_ammreg, lat_ammreg,em=em)
    #MCA spatial pattern
    U = max_cov_pattern(ssta_ammreg_3mlrc, u1000a_ammreg_3mlrc, v1000a_ammreg_3mlrc)
    #Expansion coefficients
    ntime, nrow_sst, ncol_sst = ssta_ammreg_lrc.shape
    sst2d = np.reshape(ssta_ammreg_lrc, (ntime, nrow_sst*ncol_sst), order='F')
    sstnonMissingIndex = np.where(np.isnan(sst2d[0]) == False)[0]
    sst2dNoMissing = sst2d[:, sstnonMissingIndex]

    a1 = (np.dot(sst2dNoMissing, U[:,0, np.newaxis])).squeeze()
    a1_n = ((a1.squeeze() - a1.squeeze().mean())/ a1.squeeze().std())
    return a1, a1_n

def calc_amm_index(ssta, lon, lat, em=True):
    """
    AMM index calculated using regional differences
    """
    #Northern tropical Atlantic
    lev = []
    lon1 = -60%360
    lon2 = -20%360
    lat1 = 5
    lat2 = 28
    nlons,nlats,nlevs,nssta = selreg(ssta,lon,lat,lev,lon1,lon2,lat1,lat2,\
    lev1=-999.,lev2=-999.,em=em)
    nsstam = wgtvar(ma.masked_invalid(nssta),ma.masked_invalid(nlons),\
    ma.masked_invalid(nlats),nlevs,em=em)

    #Southern tropical Atlantic
    lon1=-30%360
    lon2=360+10
    lat1=-25
    lat2=0
    slons,slats,slevs,sssta = selreg(ssta,lon,lat,lev,lon1,lon2,lat1,lat2,\
    lev1=-999.,lev2=-999.,em=em)
    ssstam = wgtvar(ma.masked_invalid(sssta),ma.masked_invalid(slons),\
    ma.masked_invalid(slats),slevs,em=em)

    #Difference
    amm = nsstam - ssstam

    return amm
#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp') #Mclm, Oclm, cmip5, obs
    args = parser.parse_args()

    iexp = str(args.exp)

    #Load detrended tropical SST anomalies from klepto
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    ssta = klepto_atm_detrend_trop[iexp+'_sst_anom']
    if iexp=='obs':
        lon = klepto_atm_detrend_trop['obs_lon']
        lat = klepto_atm_detrend_trop['obs_lat']
    else:
        lon = klepto_atm_detrend_trop['lon']
        lat = klepto_atm_detrend_trop['lat']

    if (iexp=='Mclm') or (iexp=='Oclm') or (iexp=='cmip5'):
        amm = calc_amm_index(ssta,lon,lat,em=True)
    elif (iexp=='obs'):
        amm = calc_amm_index(ssta,lon,lat,em=False)

    #Save to klepto
    klepto_indices = klepto.archives.dir_archive('klepto_indices',\
    serialized=True, cached=False)
    klepto_indices[iexp+'_amm']=amm
