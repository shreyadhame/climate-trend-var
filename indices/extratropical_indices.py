
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Atlantic Nino index"
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

def calc_nao_index(slpa, lon, lat, em=True):
    lon1  = -90%360
    lon2 = 360
    lon3 = 0
    lon4 = 40
    lat1 = 20
    lat2 = 80
    lev=[]

    lon_ta=np.concatenate((lon.sel(longitude=slice(lon1,lon2)),lon.sel(longitude=slice(lon3,lon4))+359))
    lat_ta=lat.sel(latitude=slice(lat1,lat2))

    slpa_ta1 = selreg(slpa,lon,lat,lev,lon1,lon2,lat1,lat2,lev1=[],lev2=[],em=em)[3]
    slpa_ta2 = selreg(slpa,lon,lat,lev,lon3,lon4,lat1,lat2,lev1=[],lev2=[],em=em)[3]
    slpa_ta = np.concatenate((slpa_ta1,slpa_ta2),axis=-1)

    # #PC1 and PC2
    if em==True:
        slpa_pc=np.stack([calc_eof(slpa_ta[i], lat_ta, 1, eoftype='cov')[1] for i in range(len(slpa_ta))])
        slpa_eof=np.stack([calc_eof(slpa_ta[i], lat_ta, 1, eoftype='cov')[0] for i in range(len(slpa_ta))])
        #PCs of positive phase
        for i in range(len(slpa_eof)):
            if np.nanmean(slpa_eof[i]) < 0:
                slpa_pc[i] = slpa_pc[i]*(-1.)
            else:
                pass
        slpa_pc_n = np.stack([((v - v.mean())/ v.std()) for v in slpa_pc.squeeze()])
    else:
        slpa_pc=calc_eof(slpa_ta, lat_ta, 1, eoftype='cov')[1]
        slpa_eof=calc_eof(slpa_ta, lat_ta, 1, eoftype='cov')[0]
        if np.nanmean(slpa_eof) < 0:
            slpa_pc = slpa_pc*(-1.)
        else:
            pass
        slpa_pc_n = ((slpa_pc.squeeze() - slpa_pc.squeeze().mean())/ slpa_pc.squeeze().std())
    return slpa_pc.squeeze(), slpa_pc_n

def calc_sam_index(z500a, lon, lat, em=True):
    lon1  = 0
    lon2 = 360
    lat1 = -90
    lat2 = -20
    lev=[]

    lon_ta=lon.sel(longitude=slice(lon1,lon2))
    lat_ta=lat.sel(latitude=slice(lat1,lat2))

    z500a_ta = selreg(z500a,lon,lat,lev,lon1,lon2,lat1,lat2,lev1=[],lev2=[],em=em)[3]

    # #PC1 and PC2
    if em==True:
        z500a_pc=np.stack([calc_eof(z500a_ta[i], lat_ta, 1, eoftype='cov')[1] for i in range(len(z500a_ta))])
        z500a_eof=np.stack([calc_eof(z500a_ta[i], lat_ta, 1, eoftype='cov')[0] for i in range(len(z500a_ta))])
        #PCs of positive phase
        for i in range(len(z500a_eof)):
            if np.nanmean(z500a_eof[i]) < 0:
                z500a_pc[i] = z500a_pc[i]*(-1.)
            else:
                pass
        z500a_pc_n = np.stack([((v - v.mean())/ v.std()) for v in z500a_pc.squeeze()])
    else:
        z500a_pc=calc_eof(z500a_ta, lat_ta, 1, eoftype='cov')[1]
        z500a_eof=calc_eof(z500a_ta, lat_ta, 1, eoftype='cov')[0]
        if np.nanmean(z500a_eof) < 0:
            z500a_pc = z500a_pc*(-1.)
        else:
            pass
        z500a_pc_n = ((z500a_pc.squeeze() - z500a_pc.squeeze().mean())/ z500a_pc.squeeze().std())
    return z500a_pc.squeeze(), z500a_pc_n

def calc_pna_index(z500a, lon, lat, n=1, em=True):
    lon1  = 0
    lon2 = 360
    lat1 = 20
    lat2 = 90
    lev=[]

    lon_ta=lon.sel(longitude=slice(lon1,lon2))
    lat_ta=lat.sel(latitude=slice(lat1,lat2))

    z500a_ta = selreg(z500a,lon,lat,lev,lon1,lon2,lat1-n,lat2,lev1=[],lev2=[],em=em)[3]

    # #PC1 and PC2
    if em==True:
        z500a_pc=np.stack([calc_eof(z500a_ta[i], lat_ta, 2, eoftype='cov')[1] for i in range(len(z500a_ta))])
        z500a_eof=np.stack([calc_eof(z500a_ta[i], lat_ta, 2, eoftype='cov')[0] for i in range(len(z500a_ta))])
        #Reshape to nxspace and nxtime
        z500a_pc=np.reshape(z500a_pc, (z500a_pc.shape[0], z500a_pc.shape[2], z500a_pc.shape[1]))
        z500a_eof=np.reshape(z500a_eof, (z500a_eof.shape[0], z500a_eof.shape[1],\
        z500a_eof.shape[2]*z500a_eof.shape[3]))
        #Perform varimax rotation
        z500a_pcr = np.stack([_varimax_kernel(v) for v in z500a_pc])
        z500a_eofr = np.stack([_varimax_kernel(np.nan_to_num(v)) for v in z500a_eof])
        #Reshape and select 2nd EOF
        z500a_pcr = z500a_pcr[:,1]
        z500a_eofr = z500a_eofr[:,1]
        z500a_eofr = np.reshape(z500a_eofr, (z500a_ta.shape[0],\
        z500a_ta.shape[2], z500a_ta.shape[3]))
        #PCs of positive phase
        for i in range(len(z500a_eofr)):
            if np.nanmean(z500a_eofr[i]) < 0:
                z500a_pcr[i] = z500a_pcr[i]*(-1.)
            else:
                pass
        z500a_pcr_n = np.stack([((v - v.mean())/ v.std()) for v in z500a_pcr.squeeze()])

    else:
        z500a_pc=calc_eof(z500a_ta, lat_ta, 2, eoftype='cov')[1]
        z500a_eof=calc_eof(z500a_ta, lat_ta, 2, eoftype='cov')[0]
        #Reshape to nxspace and nxtime
        z500a_pc = z500a_pc.T
        z500a_eof = np.reshape(z500a_eof, (z500a_eof.shape[0],\
        z500a_eof.shape[1]*z500a_eof.shape[2]))
        #Perform varimax rotation
        z500a_pcr = _varimax_kernel(z500a_pc)
        z500a_eofr = _varimax_kernel(z500a_eof)
        #Reshape and select 2nd EOF
        z500a_pcr = z500a_pcr[1]
        z500a_eofr = z500a_eofr[1]
        z500a_eofr = np.reshape(z500a_eofr, (z500a_ta.shape[1], z500a_ta.shape[2]))
        #PCs of positive value
        if np.nanmean(z500a_eofr) < 0:
            z500a_pcr = z500a_pcr*(-1.)
        else:
            pass
        z500a_pcr_n = ((z500a_pcr.squeeze() - z500a_pcr.squeeze().mean())/ z500a_pcr.squeeze().std())
    return z500a_pcr.squeeze(), z500a_pcr_n

def calc_psa_index(z500a, lon, lat, em=True):
    lon1  = 0
    lon2 = 360
    lat1 = -90
    lat2 = -20
    lev=[]

    lon_ta=lon.sel(longitude=slice(lon1,lon2))
    lat_ta=lat.sel(latitude=slice(lat1,lat2))

    z500a_ta = selreg(z500a,lon,lat,lev,lon1,lon2,lat1,lat2,lev1=[],lev2=[],em=em)[3]

    # #PC1 and PC2
    if em==True:
        z500a_pc=np.stack([calc_eof(z500a_ta[i], lat_ta, 2, eoftype='cov')[1] for i in range(len(z500a_ta))])
        z500a_eof=np.stack([calc_eof(z500a_ta[i], lat_ta, 2, eoftype='cov')[0] for i in range(len(z500a_ta))])
        #Reshape to nxspace and nxtime
        z500a_pc=np.reshape(z500a_pc, (z500a_pc.shape[0], z500a_pc.shape[2], z500a_pc.shape[1]))
        z500a_eof=np.reshape(z500a_eof, (z500a_eof.shape[0], z500a_eof.shape[1],\
        z500a_eof.shape[2]*z500a_eof.shape[3]))
        #Perform varimax rotation
        z500a_pcr = np.stack([_varimax_kernel(v) for v in z500a_pc])
        z500a_eofr = np.stack([_varimax_kernel(np.nan_to_num(v)) for v in z500a_eof])
        #Reshape and select 2nd EOF
        z500a_pcr = z500a_pcr[:,1]
        z500a_eofr = z500a_eofr[:,1]
        z500a_eofr = np.reshape(z500a_eofr, (z500a_ta.shape[0],\
        z500a_ta.shape[2], z500a_ta.shape[3]))
        #PCs of positive phase
        for i in range(len(z500a_eofr)):
            if np.nanmean(z500a_eofr[i]) < 0:
                z500a_pcr[i] = z500a_pcr[i]*(-1.)
            else:
                pass
        z500a_pcr_n = np.stack([((v - v.mean())/ v.std()) for v in z500a_pcr.squeeze()])

    else:
        z500a_pc=calc_eof(z500a_ta, lat_ta, 2, eoftype='cov')[1]
        z500a_eof=calc_eof(z500a_ta, lat_ta, 2, eoftype='cov')[0]
        #Reshape to nxspace and nxtime
        z500a_pc = z500a_pc.T
        z500a_eof = np.reshape(z500a_eof, (z500a_eof.shape[0],\
        z500a_eof.shape[1]*z500a_eof.shape[2]))
        #Perform varimax rotation
        z500a_pcr = _varimax_kernel(z500a_pc)
        z500a_eofr = _varimax_kernel(z500a_eof)
        #Reshape and select 2nd EOF
        z500a_pcr = z500a_pcr[1]
        z500a_eofr = z500a_eofr[1]
        z500a_eofr = np.reshape(z500a_eofr, (z500a_ta.shape[1], z500a_ta.shape[2]))
        #PCs of positive value
        if np.nanmean(z500a_eofr) < 0:
            z500a_pcr = z500a_pcr*(-1.)
        else:
            pass
        z500a_pcr_n = ((z500a_pcr.squeeze() - z500a_pcr.squeeze().mean())/ z500a_pcr.squeeze().std())
    return z500a_pcr.squeeze(), z500a_pcr_n
