#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Detrend 3D data and save to klepto_atm_data_detrended"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
import dask.array as da
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')
import klepto
import numpy as np
import numpy.ma as ma
from scipy import stats,signal
from functools import partial
import multiprocessing

#=================================================================
def func_detrend(var):
    """
    Detrends along time axis (axis=1)
    var : 4D array (em,time,lat,lon)
    """
    #Replace nans with -999999
    var = da.nan_to_num(var,-999999.)
    #Detrend
    var_detrend = signal.detrend(var,axis=1)
    #Mask zeros
    var_detrend_ma = ma.masked_where(var == 0., var_detrend) #mask
    return var_detrend_ma

#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var') #sst
    args = parser.parse_args()

    ivar = str(args.var)

    #Load data from klepto_atm_xr
    klepto_atm_data_xr = klepto.archives.dir_archive('klepto_atm_data_xr', serialized=True, cached=False)
    ds = klepto_ocean_data_xr[ivar]

    #Stack list of arrays to 4D
    ds_s = da.stack(ds,axis=0)

    #Detrend
    ds_d = func_detrend(ds_s)

    #Save to klepto
    klepto_atm_data_detrended = klepto.archives.dir_archive('klepto_atm_data_detrended', serialized=True, cached=False)
    klepto_atm_data_detrended[ivar] = ds_d
