#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Select tropical region and save to klepto_atm_detrend_trop"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')

import argparse
import dask.array as da
import klepto
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

#My modules
from read import selreg
#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var') #W,u,v,pot_temp,rho, sst, swflx
    parser.add_argument('exp') #Mclm or Oclm
    parser.add_argument('type') #mean, clim, anom
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)
    itype = str(args.type)

    #Load detrended data (mean,clim,anom)
    klepto_atm_detrend = klepto.archives.dir_archive('klepto_atm_detrend', serialized=True, cached=False)
    var = klepto_atm_detrend[iexp+'_'+ivar+'_'+itype]
    if iexp=='obs':
        lon = klepto_atm_detrend['obs_lon']
        lat = klepto_atm_detrend['obs_lat']
        lev = klepto_atm_detrend['obs_lev']
        time = klepto_atm_detrend['time']
    else:
        lon = klepto_atm_detrend['lon']
        lat = klepto_atm_detrend['lat']
        lev = klepto_atm_detrend['lev']
        time = klepto_atm_detrend['time']

    #Tropical region
    lon1_reg1=110
    lon2_reg1=360
    lat1=-60
    lat2=60
    lev1=0
    lev2=300

    lon1_reg2=0
    lon2_reg2=15

    if type(var)!=list: #3D
        #Select region 1
        lon_reg1,lat_reg,lev_reg,var_reg1 = selreg(var,lon,lat,lev,\
        lon1_reg1,lon2_reg1,lat1,lat2,lev1=-999.,lev2=-999.,em=True)

        #Select region 2
        lon_reg2,lat_reg,lev_reg,var_reg2 = selreg(var,lon,lat,lev,\
        lon1_reg2,lon2_reg2,lat1,lat2,lev1=-999.,lev2=-999.,em=True)

        #Concatenate along lon axis
        var_reg = np.concatenate((var_reg1,var_reg2),axis=-1)
        lon_reg = np.concatenate((lon_reg1,lon_reg2))

    elif type(var)==list: #4D
        var_reg1 = [selreg(v,lon,lat,lev,\
        lon1_reg1,lon2_reg1,lat1,lat2,lev1=lev1,lev2=lev2,em=False)[-1] for v in var]
        lon_reg1=selreg(var[0],lon,lat,lev,\
        lon1_reg1,lon2_reg1,lat1,lat2,lev1=lev1,lev2=lev2,em=False)[0]

        var_reg2 = [selreg(v,lon,lat,lev,\
        lon1_reg2,lon2_reg2,lat1,lat2,lev1=lev1,lev2=lev2,em=False)[-1] for v in var]
        lon_reg2,lat_reg,lev_reg=selreg(var[0],lon,lat,lev,\
        lon1_reg2,lon2_reg2,lat1,lat2,lev1=lev1,lev2=lev2,em=False)[:-1]

        #Concatenate along lon axis
        var_reg = [da.concatenate((var_reg1[i],var_reg2[i]),axis=-1) \
        for i in range(len(var_reg1))]
        lon_reg = np.concatenate((lon_reg1,lon_reg2))

        var_reg = np.stack(var_reg)

    #Add 360 after lon360
    for i in range(len(lon_reg)):
        if lon_reg[i] < 15.:
            lon_reg[i] = lon_reg[i]+360.
        else:
            pass

    #Save to klepto_atm_detrend_trop
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    klepto_atm_detrend_trop[iexp+'_'+ivar+'_'+itype+'_60'] = var_reg
    klepto_atm_detrend_trop['lon'] = lon_reg
    klepto_atm_detrend_trop['lat_60'] = lat_reg
    # klepto_atm_detrend_trop['lev'] = lev_reg
