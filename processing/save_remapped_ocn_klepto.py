#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Save remapped ocean data to klepto_atm_xr"
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
    parser.add_argument('var') #sst
    parser.add_argument('exp') #Mclm or Oclm
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)

    main_path = '/srv/ccrc/data25/z5166746/IOtrendX_pm/ocn/monthly/raw_data/'

    if ivar=='W':
        #Load W data
        clm1=main_path+'a10_'+iexp+'_r01/a10_'+iexp+'_r01.mocn_1951-2016_w_r.nc'
        clm2=main_path+'a10_'+iexp+'_r02/a10_'+iexp+'_r02.mocn_1951-2016_w_r.nc'
        clm3=main_path+'a10_'+iexp+'_r03/a10_'+iexp+'_r03.mocn_1951-2016_w_r.nc'
        clm4=main_path+'a10_'+iexp+'_r04/a10_'+iexp+'_r04.mocn_1951-2016_w_r.nc'
        clm5=main_path+'a10_'+iexp+'_r05/a10_'+iexp+'_r05.mocn_1951-2016_w_r.nc'
        clm6=main_path+'a10_'+iexp+'_r06/a10_'+iexp+'_r06.mocn_1951-2016_w_r.nc'

        clmt1=main_path+'a10_'+iexp+'T_r01/a10_'+iexp+'T_r01.mocn_1951-2016_w_r.nc'
        clmt2=main_path+'a10_'+iexp+'T_r02/a10_'+iexp+'T_r02.mocn_1951-2016_w_r.nc'
        clmt3=main_path+'a10_'+iexp+'T_r03/a10_'+iexp+'T_r03.mocn_1951-2016_w_r.nc'
        clmt4=main_path+'a10_'+iexp+'T_r04/a10_'+iexp+'T_r04.mocn_1951-2016_w_r.nc'
        clmt5=main_path+'a10_'+iexp+'T_r05/a10_'+iexp+'T_r05.mocn_1951-2016_w_r.nc'
        clmt6=main_path+'a10_'+iexp+'T_r06/a10_'+iexp+'T_r06.mocn_1951-2016_w_r.nc'

        w_all = [clm1, clm2, clm3, clm4, clm5, clm6, \
        clmt1, clmt2, clmt3, clmt4, clmt5, clmt6]

        #Vertical current
        ds = []
        for i in range(len(w_all)):
            lon,lat,lev,w,time,basin_mask = read_data(w_all[i],ivar,imask=None)
            w = w.sel(lev=slice(0,600))
            lev = lev.sel(lev=slice(0,600)) #Select upper ocean
            ds.append(w)
        # #Assign correct lat/lon to MclmT r02 and r03
        # ds[7]['latitude'] = ds[0].latitude
        # ds[8]['latitude'] = ds[0].latitude

    elif ivar=='ssh':
        clm1=main_path+'a10_'+iexp+'_r01/a10_'+iexp+'_r01.mocn_1951-2016_sshr.nc'
        clm2=main_path+'a10_'+iexp+'_r02/a10_'+iexp+'_r02.mocn_1951-2016_sshr.nc'
        clm3=main_path+'a10_'+iexp+'_r03/a10_'+iexp+'_r03.mocn_1951-2016_sshr.nc'
        clm4=main_path+'a10_'+iexp+'_r04/a10_'+iexp+'_r04.mocn_1951-2016_sshr.nc'
        clm5=main_path+'a10_'+iexp+'_r05/a10_'+iexp+'_r05.mocn_1951-2016_sshr.nc'
        clm6=main_path+'a10_'+iexp+'_r06/a10_'+iexp+'_r06.mocn_1951-2016_sshr.nc'

        clmt1=main_path+'a10_'+iexp+'T_r01/a10_'+iexp+'T_r01.mocn_1951-2016_sshr.nc'
        clmt2=main_path+'a10_'+iexp+'T_r02/a10_'+iexp+'T_r02.mocn_1951-2016_sshr.nc'
        clmt3=main_path+'a10_'+iexp+'T_r03/a10_'+iexp+'T_r03.mocn_1951-2016_sshr.nc'
        clmt4=main_path+'a10_'+iexp+'T_r04/a10_'+iexp+'T_r04.mocn_1951-2016_sshr.nc'
        clmt5=main_path+'a10_'+iexp+'T_r05/a10_'+iexp+'T_r05.mocn_1951-2016_sshr.nc'
        clmt6=main_path+'a10_'+iexp+'T_r06/a10_'+iexp+'T_r06.mocn_1951-2016_sshr.nc'

        ssh_all = [clm1, clm2, clm3, clm4, clm5, clm6, \
        clmt1, clmt2, clmt3, clmt4, clmt5, clmt6]

        #Vertical current
        ds = []
        for i in range(len(ssh_all)):
            lon,lat,lev,ssh,time,basin_mask = read_data(ssh_all[i],'sea_level',imask=None)
            ds.append(ssh)
        # #Assign correct lat/lon to MclmT r02 and r03
        # ds[7]['latitude'] = ds[0].latitude
        # ds[8]['latitude'] = ds[0].latitude

    else:
        clm1=main_path+'a10_'+iexp+'_r01/a10_'+iexp+'_r01.mocn_1951-2016_var_r.nc'
        clm2=main_path+'a10_'+iexp+'_r02/a10_'+iexp+'_r02.mocn_1951-2016_var_r.nc'
        clm3=main_path+'a10_'+iexp+'_r03/a10_'+iexp+'_r03.mocn_1951-2016_var_r.nc'
        clm4=main_path+'a10_'+iexp+'_r04/a10_'+iexp+'_r04.mocn_1951-2016_var_r.nc'
        clm5=main_path+'a10_'+iexp+'_r05/a10_'+iexp+'_r05.mocn_1951-2016_var_r.nc'
        clm6=main_path+'a10_'+iexp+'_r06/a10_'+iexp+'_r06.mocn_1951-2016_var_r.nc'

        clmt1=main_path+'a10_'+iexp+'T_r01/a10_'+iexp+'T_r01.mocn_1951-2016_var_r.nc'
        clmt2=main_path+'a10_'+iexp+'T_r02/a10_'+iexp+'T_r02.mocn_1951-2016_var_r.nc'
        clmt3=main_path+'a10_'+iexp+'T_r03/a10_'+iexp+'T_r03.mocn_1951-2016_var_r.nc'
        clmt4=main_path+'a10_'+iexp+'T_r04/a10_'+iexp+'T_r04.mocn_1951-2016_var_r.nc'
        clmt5=main_path+'a10_'+iexp+'T_r05/a10_'+iexp+'T_r05.mocn_1951-2016_var_r.nc'
        clmt6=main_path+'a10_'+iexp+'T_r06/a10_'+iexp+'T_r06.mocn_1951-2016_var_r.nc'

        ocn_all = [clm1, clm2, clm3, clm4, clm5, clm6, \
                 clmt1, clmt2, clmt3, clmt4, clmt5, clmt6]

        #Read data
        ds = []
        for i in range(len(ocn_all)):
            lon,lat,lev,v,time,basin_mask = read_data(ocn_all[i],ivar,imask=None)
            ds.append(v)

        #Convert SST to degrees celsius
        if ivar=='sst':
            ds = [ds[i] - 273.15 for i in range(len(ds))]
        else:
            pass

        #Select level for 4D data
        if (ivar=='u') or (ivar=='v') or \
        (ivar=='pot_temp') or (ivar=='salt') or (ivar=='rho'):
            ds = [ds[i].sel(lev=slice(0,600)) for i in range(len(ds))]
            lev = lev.sel(lev=slice(0,600)) #Select upper ocean
        else:
            pass

        # #Reverse sign for heat fluxes DO NOT REVERSE
        # if (ivar=='lw_heat') or (ivar=='evap_heat') or (ivar=='sens_heat') or \
        # (ivar=='swflx'):
        #     ds = [ds[i]*(-1.) for i in range(len(ds))]
        # else:
        #     pass

    #Save to klepto
    klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
    klepto_atm_xr[iexp+'_'+ivar] = ds
    # klepto_atm_xr['lon'] = lon
    # klepto_atm_xr['lat'] = lat
    # klepto_atm_xr['lev'] = lev
    # klepto_atm_xr['time'] = time
