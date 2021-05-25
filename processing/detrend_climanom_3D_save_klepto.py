#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Detrend 3D data and save to klepto_atm__detrend"
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
from scipy import stats,signal
from functools import partial
import multiprocessing

from climanom import clim_anom
#=================================================================
def func_detrend(var,axis=1):
    """
    Detrends along time axis (axis=1)
    var : 4D array (em,time,lat,lon)
    """
    #Replace nans with -999999
    var = da.nan_to_num(var,-999999.)
    #Detrend
    var_detrend = signal.detrend(var,axis=axis)
    #Mask zeros
    var_detrend_ma = ma.masked_where(var == 0., var_detrend) #mask
    return var_detrend_ma

def calc_intercept(y,a=0.10):
        mask=ma.masked_invalid(y)
        return stats.linregress(np.arange(len(mask)),mask)[1]

def chunks(matrix):
    """
    Converts 3D xarray into list of 1D arrays
    """
    #matrix_re = matrix.stack(z=(d1,d2))
    mat = np.array(matrix)
    matrix_re = np.reshape(mat,(mat.shape[0],mat.shape[1]*mat.shape[2]))
    matrix_chunks=np.stack(matrix_re.T, axis=0)
    return matrix_chunks

def f_mp(f, iterable, ncores):
    """
    Multiprocessing function
    """
    pool = multiprocessing.Pool(processes=ncores)
    func = partial(f)
    result = pool.map(func,iterable)
    pool.close()
    pool.join()
    return result

def calc_func(mat,f,ncores):
    matrix_chunks = chunks(mat)
    func_ = f_mp(f,matrix_chunks,ncores)
    return func_

def reshape_arr(var,ref_var): #(time,merged)
    if ref_var.ndim==4:
        #Reshape to xr shape (ndim==4)
        var_is = da.stack(var)
        var_intcpt = np.reshape(var_is,(ref_var.shape[0],ref_var.shape[-2],ref_var.shape[-1]))
        var_intcptr = np.swapaxes(np.tile(var_intcpt,(ref_var.shape[1],1,1,1)),0,1)
    elif ref_var.ndim==3:
        var_is = da.stack(var)
        var_intcpt = np.reshape(var_is,(ref_var.shape[-2],ref_var.shape[-1]))
        var_intcptr = np.tile(var_intcpt,(ref_var.shape[0],1,1))
    return var_intcptr
#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var') #sst
    parser.add_argument('exp') #Mclm
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)

    #Load data from klepto_atm_xr
    klepto_atm_xr = klepto.archives.dir_archive('klepto_atm_xr', serialized=True, cached=False)
    ds = klepto_atm_xr[iexp+'_'+ivar]
    if (ivar=='u1000') or (ivar=='v1000'):
        ds = [ma.masked_where(v>1e36,v) for v in ds]
    else:
        pass

    if iexp=='obs':
        #Detrend
        ds_d = func_detrend(ds,axis=0)
        #Calculate intercept
        ds_i = calc_func(ds,calc_intercept,ncores=3)
        #convert list to numpy array
        ds_in = [np.asarray(ds_i[i]) for i in range(len(ds_i))]
        #Reshape to xr shape (ndim==4)
        ds_ir = reshape_arr(ds_in,ds)
        #Mask (u,v)
        # ds_ir_ma = ma.masked_where(ds_ir > 1e30, ds_ir)
        # ds_ir_ma = ma.masked_where(ds_ir_ma < -1e30, ds_ir_ma)
        #Add intercept and detrended array
        ds_dt = ma.masked_invalid(ds_ir) + ds_d
        #Calculate climatology and anomaly
        clim,anom = clim_anom(ds_dt,start=None,em=False)

    else:
        #Stack list of arrays to 4D (mean)
        ds_s = np.stack(ds,axis=0)

        #Detrend
        ds_d = func_detrend(ds_s)

        #Calculate intercept
        ds_i = [calc_func(ds_s[i],calc_intercept,ncores=3)\
        for i in range(len(ds_s))]
        #convert list to numpy array
        ds_in = [np.asarray(ds_i[i]) for i in range(len(ds_i))]
        #Reshape to xr shape (ndim==4)
        ds_ir = reshape_arr(ds_in,ds_s)
        #Mask (u,v)
        # ds_ir_ma = ma.masked_where(ds_ir > 1e30, ds_ir)
        # ds_ir_ma = ma.masked_where(ds_ir_ma < -1e30, ds_ir_ma)

        #Add intercept and detrended array
        ds_dt = ma.masked_invalid(ds_ir) + ds_d

        #Calculate climatology and anomaly
        clim,anom = clim_anom(ds_dt,start=None,em=True)

    #Save to klepto
    klepto_atm_detrend = klepto.archives.dir_archive('klepto_atm_detrend', serialized=True, cached=False)
    klepto_atm_detrend[iexp+'_'+ivar+'_mean'] = ds_dt
    klepto_atm_detrend[iexp+'_'+ivar+'_clim'] = clim
    klepto_atm_detrend[iexp+'_'+ivar+'_anom'] = anom
