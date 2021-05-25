#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Calculate vertical velocity using continuity equation"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
#General modules
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')

import argparse
import klepto
import numpy as np
import numpy.ma as ma
import warnings
warnings.filterwarnings("ignore")

from climanom import clim_anom
#============================================================
def diff_axis(var,axis=-1):
    vard = np.diff(var,axis=axis)
    if vard.ndim==1:
        vard = np.insert(vard,0,[vard[0]],axis=0)
    elif vard.ndim==2:
        if axis==-1:
            vard = np.insert(vard,0,[vard[:,0]],axis=axis)
        elif axis==-2:
            vard = np.insert(vard,0,[vard[0,:]],axis=axis)
    elif vard.ndim==3:
        if axis==-1:
            vard = np.insert(vard,0,[vard[:,:,0]],axis=axis)
        elif axis==-2:
            vard = np.insert(vard,0,[vard[:,0,:]],axis=axis)
        elif axis==-3:
            vard = np.insert(vard,0,[vard[0,:,:]],axis=axis)
    elif vard.ndim==4:
        if axis==-1:
            vard = np.insert(vard,0,[vard[:,:,:,0]],axis=axis)
        elif axis==-2:
            vard = np.insert(vard,0,[vard[:,:,0,:]],axis=axis)
        elif axis==-3:
            vard = np.insert(vard,0,[vard[:,0,:,:]],axis=axis)
        elif axis==-4:
            vard = np.insert(vard,0,[vard[0,:,:,:]],axis=axis)
    return vard

def calc_w(u,v,x,y,z):
    #Zonal gradient
    dx = diff_axis(x,axis=0)*111320
    dudx = diff_axis(u,axis=-1)/dx[np.newaxis,np.newaxis,np.newaxis,:]
    #Meridional gradient
    dy = diff_axis(y,axis=0)*111320
    dvdy = diff_axis(v,axis=-2)/dy[np.newaxis,np.newaxis,:,np.newaxis]
    dz = diff_axis(z,axis=0)
    dwdz = (-1.)*(dudx+dvdy)*dz[np.newaxis,:,np.newaxis,np.newaxis]
    w = np.flip(np.cumsum(np.flip(dwdz,axis=-3),axis=-3),axis=-3)
    return w

#============================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp') #Mclm or Oclm, cmip5, obs
    args = parser.parse_args()

    iexp = str(args.exp)

    #Load u,v data
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    u = np.array(klepto_atm_detrend_trop[iexp+'_u_mean'])
    v = np.array(klepto_atm_detrend_trop[iexp+'_v_mean'])
    if iexp=='obs':
        x = np.array(klepto_atm_detrend_trop['obs_lon'])
        y = np.array(klepto_atm_detrend_trop['obs_lat'])
        z = np.array(klepto_atm_detrend_trop['obs_lev'])
    else:
        x = np.array(klepto_atm_detrend_trop['lon'])
        y = np.array(klepto_atm_detrend_trop['lat'])
        z = np.array(klepto_atm_detrend_trop['lev'])

    #Calculate w
    if iexp=='obs':
        wm =  calc_w(u,v,x,y,z)
        #Calculate clim and anom
        wc, wa = clim_anom(wm,start=None,em=False)
    else:
        wm =  np.stack([calc_w(u[i],v[i],x,y,z) for i in range(len(u))])
        #Calculate clim and anom
        wc, wa = clim_anom(wm,start=None,em=True)

    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    klepto_atm_detrend_trop[iexp+'_w_mean'] = wm
    klepto_atm_detrend_trop[iexp+'_w_clim'] = wc
    klepto_atm_detrend_trop[iexp+'_w_anom'] = wa
