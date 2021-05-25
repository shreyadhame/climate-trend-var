#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Save cdo detrended 4D data (u,v,w,pot_temp) to klepto_atm_data_detrended"
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
    parser.add_argument('exp') #Mclm or Oclm
    parser.add_argument('type') #clim, anom, mean
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)
    itype = str(args.type)

    #Paths to detrended files for u,v,w,pot_temp
    main_path = '/srv/ccrc/data25/z5166746/IOtrendX_pm/ocn/monthly/raw_data/'

    if ivar=='W':
        #Load W data
        clm1=main_path+'a10_'+iexp+'_r01/a10_'+iexp+'_r01.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clm2=main_path+'a10_'+iexp+'_r02/a10_'+iexp+'_r02.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clm3=main_path+'a10_'+iexp+'_r03/a10_'+iexp+'_r03.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clm4=main_path+'a10_'+iexp+'_r04/a10_'+iexp+'_r04.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clm5=main_path+'a10_'+iexp+'_r05/a10_'+iexp+'_r05.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clm6=main_path+'a10_'+iexp+'_r06/a10_'+iexp+'_r06.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'

        clmt1=main_path+'a10_'+iexp+'T_r01/a10_'+iexp+'T_r01.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clmt2=main_path+'a10_'+iexp+'T_r02/a10_'+iexp+'T_r02.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clmt3=main_path+'a10_'+iexp+'T_r03/a10_'+iexp+'T_r03.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clmt4=main_path+'a10_'+iexp+'T_r04/a10_'+iexp+'T_r04.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clmt5=main_path+'a10_'+iexp+'T_r05/a10_'+iexp+'T_r05.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'
        clmt6=main_path+'a10_'+iexp+'T_r06/a10_'+iexp+'T_r06.mocn_1951-2016_w_r_detrend_'+str(itype)+'.nc'

        atm_all = [clm1, clm2, clm3, clm4, clm5, clm6, clmt1, clmt2, clmt3, clmt4, clmt5, clmt6]

        #Vertical current
        iw = 'W'
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,w,time,basin_mask = read_data(atm_all[i],iw,imask=None)
            ds.append(w)
        # #Assign correct lat/lon to MclmT r02 and r03
        # w_d[7]['latitude'] = w_d[0].latitude
        # w_d[8]['latitude'] = w_d[0].latitude

    else:
        #u
        clm1=main_path+'a10_'+iexp+'_r01/a10_'+iexp+'_r01.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clm2=main_path+'a10_'+iexp+'_r02/a10_'+iexp+'_r02.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clm3=main_path+'a10_'+iexp+'_r03/a10_'+iexp+'_r03.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clm4=main_path+'a10_'+iexp+'_r04/a10_'+iexp+'_r04.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clm5=main_path+'a10_'+iexp+'_r05/a10_'+iexp+'_r05.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clm6=main_path+'a10_'+iexp+'_r06/a10_'+iexp+'_r06.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'

        clmt1=main_path+'a10_'+iexp+'T_r01/a10_'+iexp+'T_r01.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clmt2=main_path+'a10_'+iexp+'T_r02/a10_'+iexp+'T_r02.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clmt3=main_path+'a10_'+iexp+'T_r03/a10_'+iexp+'T_r03.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clmt4=main_path+'a10_'+iexp+'T_r04/a10_'+iexp+'T_r04.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clmt5=main_path+'a10_'+iexp+'T_r05/a10_'+iexp+'T_r05.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'
        clmt6=main_path+'a10_'+iexp+'T_r06/a10_'+iexp+'T_r06.mocn_1951-2016_uvpot_temprho_detrend_'+str(itype)+'.nc'

        atm_all = [clm1, clm2, clm3, clm4, clm5, clm6, clmt1, clmt2, clmt3, clmt4, clmt5, clmt6]

        iv = ivar
        ds = []
        for i in range(len(atm_all)):
            lon,lat,lev,v,time,basin_mask = read_data(atm_all[i],iv,imask=None)
            ds.append(v)

    #Save to klepto
    klepto_atm_detrend = klepto.archives.dir_archive('klepto_atm_detrend', serialized=True, cached=False)
    klepto_atm_detrend[iexp+'_'+ivar+'_'+itype] = ds
    # klepto_atm_detrend['ud_iotrend'] = u_d[n:]
    # klepto_atm_detrend['vd_control'] = v_d[:n]
    # klepto_atm_detrend['vd_iotrend'] = v_d[n:]
    # klepto_atm_detrend['wd_control'] = w_d[:n]
    # klepto_atm_detrend['wd_iotrend'] = w_d[n:]
    # klepto_atm_detrend['pot_tempd_control'] = pot_temp_d[:n]
    # klepto_atm_detrend['pot_tempd_iotrend'] = pot_temp_d[n:]
