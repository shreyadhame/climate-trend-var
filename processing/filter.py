#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Filter indices and data for interannual variability"
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
import scipy.signal

#My modules
from climanom import clim_anom
#=================================================================
def butter_filter(input, cutoff, lowcut, highcut, fs, order, btype='low'):
    if btype=='low':
        #Low pass Butterworth Filter
        nyq = 0.5 * fs
        normal_cutoff = (1/cutoff) / nyq
    elif btype=='high':
        #High pass Butterworth Filter
        nyq = 0.5 * fs
        normal_cutoff = (1/cutoff) / nyq
    elif btype=='band':
        nyq = 0.5 * fs
        normal_cutoff = [(1/lowcut)/nyq,(1/highcut)/nyq] # Cutoff frequency
    # Get the filter coefficients
    B, A = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False,fs=fs)
    filtered = scipy.signal.filtfilt(B, A, input)
    return filtered

#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var')
    parser.add_argument('exp') #Mclm, Mclm, cmip5, obs
    parser.add_argument('axis') #1
    args = parser.parse_args()

    ivar = str(args.var)
    iexp = str(args.exp)
    iaxis = int(args.axis)

    #Load variables
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop', serialized=True, cached=False)
    # if iexp=='Mclm':
        # var_mean=np.array(klepto_atm_detrend_trop[str(iexp)+'_'+str(ivar)+'_mean'])[9:]
    # else:
    var_mean=np.array(klepto_atm_detrend_trop[str(iexp)+'_'+str(ivar)+'_mean'])
    # var_mean=np.array(klepto_atm_detrend_trop['Mclm_io2_'+str(ivar)+'_mean'])

    #Filter mean
    var_mean_f = np.apply_along_axis(lambda x: butter_filter(x, cutoff=9*12, lowcut=None, highcut=None, fs=1, order=6, \
                                                        btype='high'),iaxis,var_mean)

    if iaxis==1:
        var_mean_m = np.nanmean(var_mean,axis=iaxis)[:,np.newaxis]
        #Add mean to filtered data
        var_mean_fm = var_mean_f + var_mean_m
        var_clim_f,var_anom_f = clim_anom(var_mean_fm,start=None,em=True)
    elif iaxis==0:
        var_mean_m = np.nanmean(var_mean,axis=iaxis)[np.newaxis]
        #Add mean to filtered data
        var_mean_fm = var_mean_f + var_mean_m
        var_clim_f,var_anom_f = clim_anom(var_mean_fm,start=None,em=False)

    klepto_atm_detrend_trop_flt = klepto.archives.dir_archive('klepto_atm_detrend_trop_flt', serialized=True, cached=False)
    klepto_atm_detrend_trop_flt[str(iexp)+'_'+str(ivar)+'_anom']=var_anom_f
    klepto_atm_detrend_trop_flt[str(iexp)+'_'+str(ivar)+'_mean']=var_mean_fm
    klepto_atm_detrend_trop_flt[str(iexp)+'_'+str(ivar)+'_clim']=var_clim_f
