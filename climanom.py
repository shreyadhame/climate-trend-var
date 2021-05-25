#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Calculate climatology, anomalies"
__reference__ = "https://gcmd.gsfc.nasa.gov/KeywordSearch/Metadata.do?Portal\
=amd&KeywordPath=&OrigMetadataNode=GCMD&EntryId=Indian_Ocean_Dipole&MetadataView\
=Full&MetadataType=0&lbnode=mdlb2"
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
# import pandas as pd
from eofs.standard import Eof
import scipy
from sklearn.linear_model import TheilSenRegressor

#My modules
from read import *
#from create_forcingfile import mask_oceans
#============================================================

def clim_anom_xr(var,time,start=None,end='2017'):
    """
    Calculates the climatology and anomaly of a climate dataset

    Input: Variable (ex. SST, wind)
           Time (dates)
           Start date
           End date

    Output: Climatology
            Anomaly
    """
    #Select the years for calculating climatology
    if start == None:
        clim = var.groupby('time.month').mean(dim='time')
    else:
        var_t = var.sel(time=slice(start,end))
        clim = var_t.groupby('time.month').mean(dim='time')
    anom = var.groupby('time.month') - clim
    return clim, anom

# #Use when time is manually defined
# def clim_anom(var,time,start,end='2017-12',nem=0,nmonths=12,nmonths_rm=30*12):
#     #calculate Climatology
#     if start == None:
#         if nem > 0:
#         #create an empy numpy array of a new shape for Climatology
#             if var.ndim ==5: #(em,time,lev,lat,lon)
#                 clim = np.zeros((nem,nmonths,var.shape[-3],var.shape[-2],\
#                 var.shape[-1]),dtype=float)
#                 for i in np.arange(0,nmonths):
#                     clim[:,i,:,:,:] = np.nanmean(var[:,i:-1:nmonths,:,:,:], axis=1)
#                 clim = np.tile(clim,(1,len(var[0])//nmonths,1,1,1))
#             elif var.ndim ==4: #(em,time,lat,lon)
#                 clim = np.zeros((nem,nmonths,var.shape[-2],var.shape[-1]),dtype=float)
#                 for i in np.arange(0,nmonths):
#                     clim[:,i,:,:] = np.nanmean(var[:,i:-1:nmonths,:,:], axis=1)
#                 clim = np.tile(clim,(1,len(var[0])//nmonths,1,1))
#         elif nem == 0:
#             if var.ndim ==4: #(time,lev,lat,lon)
#                 clim = np.zeros((nmonths,var.shape[-3],var.shape[-2],var.shape[-1]),dtype=float)
#                 for i in np.arange(0,nmonths):
#                     clim[i,:,:,:] = np.nanmean(var[i:-1:nmonths,:,:,:], axis=0)
#                 clim = np.tile(clim,(len(var)//nmonths,1,1,1))
#             elif var.ndim ==3: #(time,lat,lon)
#                 clim = np.zeros((nmonths,var.shape[-2],var.shape[-1]),dtype=float)
#                 for i in np.arange(0,nmonths):
#                     clim[i,:,:] = np.nanmean(var[i:-1:nmonths,:,:], axis=0)
#                 clim = np.tile(clim,(len(var)//nmonths,1,1))
#             elif var.ndim ==2: #(time,lon/lat)
#                 clim = np.zeros((nmonths,var.shape[1]),dtype=float)
#                 for i in np.arange(0,nmonths):
#                     clim[i,:] = np.nanmean(var[i:-1:nmonths,:], axis=0)
#                 clim = np.tile(clim,(len(var)//nmonths,1))
#             elif var.ndim == 1: #(time)
#                 clim = np.zeros(nmonths,dtype=float)
#                 for i in np.arange(0,nmonths):
#                     clim[i] = np.nanmean(var[i:-1:nmonths],axis=0)
#                 clim = np.tile(clim,(len(var)//nmonths))
#     elif start == 'run':
#     #create an empy numpy array of a new shape for Climatology
#         clim = np.zeros((var.shape[0],var.shape[1],var.shape[2]),dtype=float)
#         for i in range(var.shape[0]):
#             clim[i,:,:] = np.mean(var[i:i+nmonths_rm:nmonths,:,:],axis=0)
#
#     #subtract climatology from total time series to obtain anomalies
#     anom = var - clim
#     return clim,anom
def clim_anom(var,start,end='2017-12',em=True,nmonths=12,nmonths_rm=30*12):
    #calculate Climatology:
    if start == None:
        if em==True:
            if var.ndim ==5: #(em,time,lev,lat,lon)
                var_s = np.split(var,var.shape[1]//nmonths,axis=1)
                var_d = np.stack(var_s,axis=1)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=1)
                clim = np.tile(var_m,(1,var.shape[1]//nmonths,1,1,1))
            elif var.ndim ==4: #(em,time,lat,lon)
                var_s = np.split(var,var.shape[1]//nmonths,axis=1)
                var_d = np.stack(var_s,axis=1)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=1)
                clim = np.tile(var_m,(1,var.shape[1]//nmonths,1,1))
            elif var.ndim ==2: #(em,time)
                var_s = np.split(var,var.shape[1]//nmonths,axis=1)
                var_d = np.stack(var_s,axis=1)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=1)
                clim = np.tile(var_m,(1,var.shape[1]//nmonths))
        elif em==False:
            if var.ndim ==4: #(time,lev,lat,lon)
                var_s = np.split(var,var.shape[0]//nmonths,axis=0)
                var_d = np.stack(var_s,axis=0)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=0)
                clim = np.tile(var_m,(var.shape[0]//nmonths,1,1,1))
            elif var.ndim ==3: #(time,lat,lon)
                var_s = np.split(var,var.shape[0]//nmonths,axis=0)
                var_d = np.stack(var_s,axis=0)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=0)
                clim = np.tile(var_m,(var.shape[0]//nmonths,1,1))
            elif var.ndim ==2: #(time,lon/lat)
                var_s = np.split(var,var.shape[0]//nmonths,axis=0)
                var_d = np.stack(var_s,axis=0)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=0)
                clim = np.tile(var_m,(var.shape[0]//nmonths,1))
            elif var.ndim == 1: #(time)
                var_s = np.split(var,var.shape[0]//nmonths,axis=0)
                var_d = np.stack(var_s,axis=0)
                var_m = ma.masked_where(var_d==0.,var_d).mean(axis=0)
                clim = np.tile(var_m,(var.shape[0]//nmonths))
    elif start=='run':
    #create an empy numpy array of a new shape for Climatology
        clim = np.zeros((var.shape[0],var.shape[1],var.shape[2]),dtype=float)
        for i in range(var.shape[0]):
            clim[i,:,:] = np.nanmean(var[i:i+nmonths_rm:nmonths,:,:],axis=0)

    #subtract climatology from total time series to obtain anomalies
    anom = var - clim
    return clim,anom

def normalise_index(ind):
    ind_norm = (ind - np.nanmean(ind))/np.nanstd(ind)
    return ind_norm

def movingaverage(values, window, mode='valid'):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, mode)
    return sma

# def sel_season(ssta, start, n=3, em=True):
#     if em==True:
#         if (ssta.ndim == 4) or (ssta.ndim == 2): #(em,time,lat,lon) or (em,time)
#             ssta_s = np.concatenate([ssta[:,i:i+n] for i in range(start,ssta.shape[1],12)],axis=1)
#     elif em==False:
#         if (ssta.ndim == 4): #(time,lev,lat,lon)
#             ssta_s = np.concatenate([ssta[:,i:i+n] for i in range(start,ssta.shape[1],12)],axis=0)
#         elif (ssta.ndim == 3): #(time,lat,lon)
#             ssta_s = np.concatenate([ssta[i:i+n] for i in range(start,ssta.shape[0],12)],axis=0)
#     return ssta_s

def sel_season(ssta, start, n=3, em=True):
    if em==True: #(em,time)...
        ssta_s = np.concatenate([ssta[:,i:i+n] for i in range(start,ssta.shape[1],12)],axis=1)
    elif em==False: #(time)...
        ssta_s = np.concatenate([ssta[i:i+n] for i in range(start,ssta.shape[0],12)],axis=0)
    return ssta_s

def season_mean(ind_s, n=4, em=True):
    if em==True:
        sm = np.nanmean(np.stack(np.split(ind_s, ind_s.shape[1]//n, axis=1), axis=1),axis=2)
    return sm
