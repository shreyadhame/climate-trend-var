#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Save atm data to klepto_atm_xr"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')

import argparse
import klepto
import numpy as np
import numpy.ma as ma
import xarray as xr

#My modules
from read import read_data
#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var') #u1000
    parser.add_argument('exp') #Mclm
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)

    main_path = '/srv/ccrc/data25/z5166746/IOtrendX_pm/atm/monthly/raw_data/'

    clm1=main_path+'a10_'+iexp+'_r01/a10_'+iexp+'_r01.pa_1951-2016_var.nc'
    clm2=main_path+'a10_'+iexp+'_r02/a10_'+iexp+'_r02.pa_1951-2016_var.nc'
    clm3=main_path+'a10_'+iexp+'_r03/a10_'+iexp+'_r03.pa_1951-2016_var.nc'
    clm4=main_path+'a10_'+iexp+'_r04/a10_'+iexp+'_r04.pa_1951-2016_var.nc'
    clm5=main_path+'a10_'+iexp+'_r05/a10_'+iexp+'_r05.pa_1951-2016_var.nc'
    clm6=main_path+'a10_'+iexp+'_r06/a10_'+iexp+'_r06.pa_1951-2016_var.nc'

    clmt1=main_path+'a10_'+iexp+'T_r01/a10_'+iexp+'T_r01.pa_1951-2016_var.nc'
    clmt2=main_path+'a10_'+iexp+'T_r02/a10_'+iexp+'T_r02.pa_1951-2016_var.nc'
    clmt3=main_path+'a10_'+iexp+'T_r03/a10_'+iexp+'T_r03.pa_1951-2016_var.nc'
    clmt4=main_path+'a10_'+iexp+'T_r04/a10_'+iexp+'T_r04.pa_1951-2016_var.nc'
    clmt5=main_path+'a10_'+iexp+'T_r05/a10_'+iexp+'T_r05.pa_1951-2016_var.nc'
    clmt6=main_path+'a10_'+iexp+'T_r06/a10_'+iexp+'T_r06.pa_1951-2016_var.nc'

    atm_all = [clm1, clm2, clm3, clm4, clm5, clm6, \
    clmt1, clmt2, clmt3, clmt4, clmt5, clmt6]

    #U1000
    if ivar=='u1000':
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,ua,time,basin_mask = read_data(atm_all[i],'ua_plev',imask=None)
            ua = ua.sel(lev=1000,method='nearest')
            ds.append(ua)

        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr[str(iexp)+'_u1000'] = ds

    #V1000
    elif ivar=='v1000':
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,va,time,basin_mask = read_data(atm_all[i],'va_plev',imask=None)
            va = va.sel(lev=1000,method='nearest')
            ds.append(va)

        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr[str(iexp)+'_v1000'] = ds

    elif ivar=='slp':
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,slp,time,basin_mask = read_data(atm_all[i],'psl',imask=None)
            ds.append(slp)

        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr[str(iexp)+'_slp'] = ds

    elif ivar=='pr':
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,pr,time,basin_mask = read_data(atm_all[i],'pr',imask=None)
            pr = pr*86400.
            ds.append(pr)

        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr[str(iexp)+'_pr'] = ds
