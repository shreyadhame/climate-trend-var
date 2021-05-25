#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Save observation data to klepto_atm_xr"
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
    parser.add_argument('var') #thetao,
    args = parser.parse_args()

    ivar = str(args.var)

    main_path = '/srv/ccrc/data25/z5166746/Obs_data/obs_'+str(ivar)+'_r.nc'

    if ivar=='sst':
        lon,lat,lev,sst,time,basin_mask = read_data(main_path,'sst',imask=None)

        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_sst'] = sst

    elif ivar=='pot_temp':
        lon,lat,lev,pot_temp,time,basin_mask = read_data(main_path,'temp',imask=None, decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_pot_temp'] = pot_temp

    elif ivar=='slp':
        lon,lat,lev,slp,time,basin_mask = read_data(main_path,'slp',imask=None)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_slp'] = slp

    elif ivar=='zg':
        lon,lat,lev,zg,time,basin_mask = read_data(main_path,'hgt',imask=None)
        z200=zg.sel(lev=200,method='nearest')
        z500=zg.sel(lev=500,method='nearest')
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_z200'] = z200
        klepto_atm_xr['obs_z500'] = z500

    elif ivar=='u':
        lon,lat,lev,u,time,basin_mask = read_data(main_path,ivar,imask=None,decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_u'] = u

    elif ivar=='v':
        lon,lat,lev,v,time,basin_mask = read_data(main_path,ivar,imask=None,decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_v'] = v

    elif ivar=='w':
        lon,lat,lev,w,time,basin_mask = read_data(main_path,'W',imask=None,decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_w_up'] = w[:,:9]
        klepto_atm_xr['obs_w_down'] = w[:,9:]

    elif ivar=='tflux':
        lon,lat,lev,tflux,time,basin_mask = read_data(main_path,'qnet',imask=None, decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_tflux'] = tflux

    elif ivar=='ua':
        lon,lat,lev,ua,time,basin_mask = read_data(main_path,'uwnd',imask=None)
        u200 = ua.sel(lev=200,method='nearest')
        u500 = ua.sel(lev=500,method='nearest')
        u850 = ua.sel(lev=850,method='nearest')
        u1000 = ua.sel(lev=1000,method='nearest')
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_u200'] = u200
        klepto_atm_xr['obs_u500'] = u500
        klepto_atm_xr['obs_u850'] = u850
        klepto_atm_xr['obs_u1000'] = u1000

    elif ivar=='va':
        lon,lat,lev,va,time,basin_mask = read_data(main_path,'vwnd',imask=None)
        v200 = va.sel(lev=200,method='nearest')
        v500 = va.sel(lev=500,method='nearest')
        v850 = va.sel(lev=850,method='nearest')
        v1000 = va.sel(lev=1000,method='nearest')
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_v200'] = v200
        klepto_atm_xr['obs_v500'] = v500
        klepto_atm_xr['obs_v850'] = v850
        klepto_atm_xr['obs_v1000'] = v1000

    elif ivar=='tau_x':
        lon,lat,lev,tau_x,time,basin_mask = read_data(main_path,'fu',imask=None, decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_tau_x'] = tau_x

    elif ivar=='ta':
        lon,lat,lev,ta,time,basin_mask = read_data(main_path,'air',imask=None)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_t'] = ta

    elif ivar=='ssh':
        lon,lat,lev,ssh,time,basin_mask = read_data(main_path,'zeta',imask=None, decode_times=False)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['obs_ssh'] = ssh
