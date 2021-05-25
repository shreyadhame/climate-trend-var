#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Save CMIP5 data to klepto_atm_xr"
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

    main_path = '/srv/ccrc/data25/z5166746/CMIP5/'

    cmip1=main_path+str(ivar)+'/a10_cmip5_r01_'+str(ivar)+'_r.nc'
    cmip2=main_path+str(ivar)+'/a10_cmip5_r02_'+str(ivar)+'_r.nc'
    cmip3=main_path+str(ivar)+'/a10_cmip5_r03_'+str(ivar)+'_r.nc'

    cmip_all = [cmip1,cmip2,cmip3]

    #SST
    if ivar=='sst':
        ds_sst = []
        for i in range(len(cmip_all)):
            lon,lat,lev,sst,time,basin_mask = read_data(cmip_all[i],'thetao',imask=None)
            sst = sst - 273.15 #Convert SST to degrees celsius
            ds_sst.append(sst)

            klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
            klepto_atm_xr['cmip5_sst'] = ds_sst

    elif ivar=='pot_temp':
        ds_pot_temp = []
        for i in range(len(cmip_all)):
            lon,lat,lev,pot_temp,time,basin_mask = read_data(cmip_all[i],'thetao',imask=None)
            pot_temp = pot_temp.sel(lev=slice(0,600)) - 273.15 #Convert SST to degrees celsius
            lev = lev.sel(lev=slice(0,600)) #Select upper ocean
            ds_pot_temp.append(pot_temp)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_pot_temp'] = ds_pot_temp

    elif ivar=='slp':
        ds_slp = []
        for i in range(len(cmip_all)):
            lon,lat,lev,slp,time,basin_mask = read_data(cmip_all[i],'psl',imask=None)
            ds_slp.append(slp)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_slp'] = ds_slp

    elif ivar=='zg':
        ds_z200 = []
        ds_z500 = []
        for i in range(len(cmip_all)):
            lon,lat,lev,zg,time,basin_mask = read_data(cmip_all[i],'zg',imask=None)
            z200 = zg.sel(lev=20000,method='nearest')
            z500 = zg.sel(lev=50000,method='nearest')
            ds_z200.append(z200)
            ds_z500.append(z500)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_z200'] = ds_z200
        klepto_atm_xr['cmip5_z500'] = ds_z500

    elif ivar=='u':
        ds_u = []
        for i in range(len(cmip_all)):
            lon,lat,lev,u,time,basin_mask = read_data(cmip_all[i],'uo',imask=None)
            ds_u.append(u)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_u'] = ds_u

    elif ivar=='v':
        ds_v = []
        for i in range(len(cmip_all)):
            lon,lat,lev,v,time,basin_mask = read_data(cmip_all[i],'vo',imask=None)
            ds_v.append(v)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_v'] = ds_v

    elif ivar=='w':
        ds_w = []
        for i in range(len(cmip_all)):
            lon,lat,lev,w,time,basin_mask = read_data(cmip_all[i],'W',imask=None)
            ds_w.append(w)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_w'] = ds_w

    elif ivar=='tflux':
        ds_tflux = []
        for i in range(len(cmip_all)):
            lon,lat,lev,tflux,time,basin_mask = read_data(cmip_all[i],'hfds',imask=None)
            ds_tflux.append(tflux)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_tflux'] = ds_tflux

    elif ivar=='evap_heat':
        ds_hfls = []
        for i in range(len(cmip_all)):
            lon,lat,lev,hfls,time,basin_mask = read_data(cmip_all[i],'hfls',imask=None)
            ds_hfls.append(hfls)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_evap_heat'] = ds_hfls

    elif ivar=='ssh':
        ds_zos = []
        for i in range(len(cmip_all)):
            lon,lat,lev,zos,time,basin_mask = read_data(cmip_all[i],'zos',imask=None)
            ds_zos.append(zos)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_ssh'] = ds_zos

    elif ivar=='ua':
        ds_u200 = []
        ds_u500 = []
        ds_u850 = []
        ds_u1000 = []
        for i in range(len(cmip_all)):
            lon,lat,lev,ua,time,basin_mask = read_data(cmip_all[i],'ua',imask=None)
            u200 = ua.sel(lev=20000,method='nearest')
            u500 = ua.sel(lev=50000,method='nearest')
            u850 = ua.sel(lev=85000,method='nearest')
            u1000 = ua.sel(lev=100000,method='nearest')
            ds_u200.append(u200)
            ds_u500.append(u500)
            ds_u850.append(u850)
            ds_u1000.append(u1000)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_u200'] = ds_u200
        klepto_atm_xr['cmip5_u500'] = ds_u500
        klepto_atm_xr['cmip5_u850'] = ds_u850
        klepto_atm_xr['cmip5_u1000'] = ds_u1000

    elif ivar=='va':
        ds_v200 = []
        ds_v500 = []
        ds_v850 = []
        ds_v1000 = []
        for i in range(len(cmip_all)):
            lon,lat,lev,va,time,basin_mask = read_data(cmip_all[i],'va',imask=None)
            v200 = va.sel(lev=20000,method='nearest')
            v500 = va.sel(lev=50000,method='nearest')
            v850 = va.sel(lev=85000,method='nearest')
            v1000 = va.sel(lev=100000,method='nearest')
            ds_v200.append(v200)
            ds_v500.append(v500)
            ds_v850.append(v850)
            ds_v1000.append(v1000)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_v200'] = ds_v200
        klepto_atm_xr['cmip5_v500'] = ds_v500
        klepto_atm_xr['cmip5_v850'] = ds_v850
        klepto_atm_xr['cmip5_v1000'] = ds_v1000

    elif ivar=='tau_x':
        ds_tau_x = []
        for i in range(len(cmip_all)):
            lon,lat,lev,tau_x,time,basin_mask = read_data(cmip_all[i],'tauuo',imask=None)
            ds_tau_x.append(tau_x)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_tau_x'] = ds_tau_x

    elif ivar=='mld':
        ds_mld = []
        for i in range(len(cmip_all)):
            lon,lat,lev,mld,time,basin_mask = read_data(cmip_all[i],'mlotst',imask=None)
            ds_mld.append(mld)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_mld'] = ds_mld

    elif ivar=='rho':
        ds_rho = []
        for i in range(len(cmip_all)):
            lon,lat,lev,rho,time,basin_mask = read_data(cmip_all[i],'rhopoto',imask=None)
            ds_rho.append(rho)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_rho'] = ds_rho

    elif ivar=='ta':
        ds_ta = []
        for i in range(len(cmip_all)):
            lon,lat,lev,ta,time,basin_mask = read_data(cmip_all[i],'ta',imask=None)
            ds_ta.append(ta)
        klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
        klepto_atm_xr['cmip5_t'] = ds_ta
