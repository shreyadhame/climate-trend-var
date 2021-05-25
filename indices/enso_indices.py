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
import klepto
import numpy as np
import numpy.ma as ma
import xarray as xr
from eofs.standard import Eof
import scipy
from sklearn.linear_model import TheilSenRegressor

#My modules
from read import *
from eof_analysis import *
#============================================================

def nino_ind(sst_anom, lon, lat, lev=[], lon1=-150%360, lon2=-90%360, \
lat1=-5, lat2=5, lev1=-999., lev2=-999., em=False):
    """
    Computes the Nino index for 3D array (time,lat,lon)
    Nino3 : lon1=-150%360, lon2=-90%360
    Nino4 : lon1=160%360, lon2=-150%360
    Nino3.4 : lon1=-170%360, lon2=-120%360
    """

    lons,lats,levs,ssta_nino = selreg(sst_anom,lon,lat,lev,lon1,lon2,lat1,lat2,lev1,lev2,em)
    nino = wgtvar(ma.masked_invalid(ssta_nino),ma.masked_invalid(lons),\
    ma.masked_invalid(lats),lev,em)
    # nino_norm = (nino_3 - np.nanmean(nino_3))/np.nanstd(nino_3)
    return nino

def filter_121(signal):
    length=len(signal)
    output = np.zeros(length-2)
    coef= np.array([1,2,1])
    for i in range(length - 2):
        output[i] = np.sum(signal[i:i+3] * coef / 4)
    output = np.concatenate(([signal[0]],output,[signal[-1]]))
    return output

def calc_ec_index(ssta_ndjf, lon, lat, em=True):
    """
    Calculates E- and C- indices
    """
    #Select tropical Pacific
    lon1  = 150%360
    lon2 = -80%360
    lat1 = -10
    lat2 = 10
    lev=[]

    lon_tp, lat_tp, lev_tp, ssta_tp = selreg(ssta_ndjf,lon,lat,lev,lon1,lon2,lat1,lat2,\
    lev1=-999.,lev2=-999.,em=em)

    if em==True: #Ensemble members
        ssta_pc=np.stack([calc_eof(ssta_tp[i], lat_tp, 2, eoftype='cov')[1] for i in range(len(ssta_tp))])
        ssta_eof=np.stack([calc_eof(ssta_tp[i], lat_tp, 2, eoftype='cov')[0] for i in range(len(ssta_tp))])
        #PCs of warm phase
        for i in range(len(ssta_eof)):
            for j in range(2):
                if np.nanmean(ssta_eof[i,j]) < 0:
                    ssta_pc[i,:,j] = ssta_pc[i,:,j]*(-1.)
                else:
                    pass

        #Normalise each PC by its standard deviation
        npc1 = np.stack([ssta_pc[i,:,0]-np.nanmean(ssta_pc[i,:,0])/np.nanstd(ssta_pc[i,:,0]) for i in range(len(ssta_pc))])
        npc2 = np.stack([ssta_pc[i,:,1]-np.nanmean(ssta_pc[i,:,1])/np.nanstd(ssta_pc[i,:,1]) for i in range(len(ssta_pc))])

        #Smooth using a 121 filter
        fpc1 = np.stack([filter_121(npc1[i]) for i in range(len(ssta_pc))])
        fpc2 = np.stack([filter_121(npc2[i]) for i in range(len(ssta_pc))])

        #Calculate EP and CP index
        ep_idx = (npc1 - npc2)/np.sqrt(2)
        cp_idx = (npc1 + npc2)/np.sqrt(2)

    elif em==False:
        ssta_pc=calc_eof(ssta_tp, lat_tp, 2, eoftype='cov')[1]
        ssta_eof=calc_eof(ssta_tp, lat_tp, 2, eoftype='cov')[0]
        for j in range(2):
            if np.nanmean(ssta_eof[j]) < 0:
                ssta_pc[:,j] = ssta_pc[:,j]*(-1.)
            else:
                pass

        #Normalise each PC by its standard deviation
        npc1 = ssta_pc[:,0]-np.nanmean(ssta_pc[:,0])/np.nanstd(ssta_pc[:,0])
        npc2 = ssta_pc[:,1]-np.nanmean(ssta_pc[:,1])/np.nanstd(ssta_pc[:,1])

        #Smooth using a 121 filter
        fpc1 = filter_121(npc1)
        fpc2 = filter_121(npc2)

        #Calculate EP and CP index
        ep_idx = (npc1 - npc2)/np.sqrt(2)
        cp_idx = (npc1 + npc2)/np.sqrt(2)

    return ep_idx, cp_idx

def calc_nct_nwp_index(nino3, nino4):
    a = np.zeros(len(nino3))
    a[np.where(nino3*nino4 > 0.)] = 2/5
    nct = nino3 - (a*nino4)
    nwp = nino4 - (a*nino3)
    return nct, nwp


# def count_ep_cp_elnino(e_idx, c_idx): #(time)
#     x = 1.
#     idx_ep = []
#     for i in range(len(e_idx)):
#         if (e_idx[i]>x):
#             idx_ep.append([i])
#     idx_ep = [item for sublist in idx_ep for item in sublist]
#     count_ep = len(idx_ep)
#
#     idx_cp = []
#     for i in range(len(c_idx)):
#         if (c_idx[i]>x):
#             idx_cp.append([i])
#     idx_cp = [item for sublist in idx_cp for item in sublist]
#     count_cp = len(idx_cp)
#
#     return count_ep, count_cp, idx_ep, idx_cp
#
# def count_ep_cp_lanina(e_idx, c_idx): #(time)
#     x = -1.
#     idx_ep = []
#     for i in range(len(e_idx)):
#         if (e_idx[i]<x):
#             idx_ep.append([i])
#     idx_ep = [item for sublist in idx_ep for item in sublist]
#     count_ep = len(idx_ep)
#
#     idx_cp = []
#     for i in range(len(c_idx)):
#         if (c_idx[i]<x):
#             idx_cp.append([i])
#     idx_cp = [item for sublist in idx_cp for item in sublist]
#     count_cp = len(idx_cp)
#
#     return count_ep, count_cp, idx_ep, idx_cp
#


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
        #NDJF mean
        #Select and average season
        ssta_ndjf = np.concatenate([ssta[i:i+4] for i in range(10,ssta.shape[0]-12,12)],axis=0)
        #Mean of season
        ssta_ndjfm = np.nanmean(np.stack(np.split(ssta_ndjf, ssta_ndjf.shape[0]//4, axis=0), axis=0),axis=1)
    else:
        lon = klepto_atm_detrend_trop['lon']
        lat = klepto_atm_detrend_trop['lat']
        #NDJF mean
        #Select and average season
        ssta_ndjf = np.concatenate([ssta[:,i:i+4] for i in range(10,ssta.shape[1]-12,12)],axis=1)
        #Mean of season
        ssta_ndjfm = np.nanmean(np.stack(np.split(ssta_ndjf, ssta_ndjf.shape[1]//4, axis=1), axis=1),axis=2)

    if (iexp=='Mclm') or (iexp=='Oclm') or (iexp=='cmip5'):
        nino3 = nino_ind(ssta,lon,lat,lev=[],lon1=-150%360,lon2=-90%360,em=True)
        nino4 = nino_ind(ssta,lon,lat,lev=[],lon1=160%360,lon2=-150%360,em=True)
        #Calculate E and C index
        e_index, c_index = calc_ec_index(ssta_ndjfm,lon,lat,em=True)
    elif (iexp=='obs'):
        nino3 = nino_ind(ssta,lon,lat,lev=[],lon1=-150%360,lon2=-90%360,em=False)
        nino4 = nino_ind(ssta,lon,lat,lev=[],lon1=160%360,lon2=-150%360,em=False)
        #Calculate Atl Nino index
        e_index, c_index = calc_ec_index(ssta_ndjfm,lon,lat,em=False)

    #Save to klepto
    klepto_indices = klepto.archives.dir_archive('klepto_indices',\
    serialized=True, cached=False)
    klepto_indices[iexp+'_nino3']=nino3
    klepto_indices[iexp+'_nino4']=nino4
    klepto_indices[iexp+'_e_index']=e_index
    klepto_indices[iexp+'_c_index']=c_index
