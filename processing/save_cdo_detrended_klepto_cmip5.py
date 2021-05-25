#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Save cdo detrended 4D data (CMIP5; u,v,w,pot_temp) to klepto_atm_data_detrended"
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

#My modules
from read import read_data
#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var') #W,u,v,pot_temp,rho
    parser.add_argument('exp') #Mcmip or Ocmip
    parser.add_argument('type') #clim, anom, mean
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)
    itype = str(args.type)

    #Paths to detrended files for u,v,w,pot_temp
    main_path = '/srv/ccrc/data25/z5166746/CMIP5/'

    if ivar=='W':
        #Load W data
        cmip1=main_path+'w/a10_'+iexp+'_r01_w_r_detrend_'+str(itype)+'.nc'
        cmip2=main_path+'w/a10_'+iexp+'_r02_w_r_detrend_'+str(itype)+'.nc'
        cmip3=main_path+'w/a10_'+iexp+'_r03_w_r_detrend_'+str(itype)+'.nc'

        atm_all = [cmip1, cmip2, cmip3]

        #Vertical current
        iw = 'W'
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,w,time,basin_mask = read_data(atm_all[i],iw,imask=None)
            ds.append(w)

    else:
        cmip1=main_path+'uvpot_temprho/a10_'+iexp+'_r01_uvpot_temprho_detrend_'+str(itype)+'.nc'
        cmip2=main_path+'uvpot_temprho/a10_'+iexp+'_r02_uvpot_temprho_detrend_'+str(itype)+'.nc'
        cmip3=main_path+'uvpot_temprho/a10_'+iexp+'_r03_uvpot_temprho_detrend_'+str(itype)+'.nc'

        atm_all = [cmip1, cmip2, cmip3]

        iv = ivar
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,v,time,basin_mask = read_data(atm_all[i],iv,imask=None)
            ds.append(v)

    #Save to klepto
    klepto_atm_detrend = klepto.archives.dir_archive('klepto_atm_detrend', serialized=True, cached=False)
    klepto_atm_detrend[iexp+'_'+ivar+'_'+itype] = ds
