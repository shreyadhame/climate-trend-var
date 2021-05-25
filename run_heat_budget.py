#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Heat budget analysis"
__reference__ = "Abellan et al. "
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
#General modules
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')

import argparse
import dask.array as da
import klepto
import gc
import numpy.ma as ma
import pandas as pd
from scipy import stats
from scipy import signal
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#My modules
from read import selreg
from climanom import normalise_index
from plots import *
from composite import *
from run_bji import diff_axis
#=================================================================
def integrate_over_ml(evar, lvar):
    evari = np.nansum(evar,axis=-1)
    lvari = np.nansum(lvar,axis=-1)
    return evari,lvari

def ave_latlon(evar,lvar):
    evarm = np.nanmean(np.nanmean(evar,axis=-1),axis=-1)
    lvarm = np.nanmean(np.nanmean(lvar,axis=-1),axis=-1)
    return evarm,lvarm

def heat_budget(pot_temp_mean, pot_temp_anom, pot_temp_clim, tflux_anom,
u_clim, u_anom, v_clim, v_anom, w_clim, w_anom,
iw_yr0, iw_yr1, ic_yr0, ic_yr1,
lon, lat, lev,
lon1, lon2, lat1, lat2, lev1, lev2, m):

    #Total heating
    #dT/dt
    dTa = diff_axis(pot_temp_anom,axis=-4)

    #Composite dT/dt
    lont, wdTa, cdTa = composite_var(dTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True,lonmean=True)

    #Integrate over the mixed layer
    iwdTa, icdTa = integrate_over_ml(wdTa,cdTa)

    del pot_temp_mean,dTa,wdTa,cdTa
    gc.collect()

    #Net surface heat flux
    rho = 1026.
    cp = 3986.
    Q = (tflux_anom/(rho*cp*lev2))*(-7.152e8) #Convert K/s to C/month

    #Composite region
    lont, wQ, cQ = composite_var(Q,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,\
        lon,lat,lev=[],\
        lon1=lon1,lon2=lon2,lat1=lat1,lat2=lat2,lev1=-999.,lev2=-999.,\
        em=False,latmean=True,lonmean=True)

    #Zonal advection
    dTadx = diff_axis(np.array(pot_temp_anom),axis=-1)/(np.diff(lon)[0]*111320.)*\
    (1/3.80517e-7)
    dTcdx = diff_axis(np.array(pot_temp_clim),axis=-1)/(np.diff(lon)[0]*111320.)*\
    (1/3.80517e-7)

    ucTa = (-1.)*u_clim*dTadx
    uaTc = (-1.)*u_anom*dTcdx
    uaTa = (-1.)*u_anom*dTadx

    #Composite region
    lont, wucTa, cucTa = composite_var(ucTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True,lonmean=True)

    lont, wuaTc, cuaTc = composite_var(uaTc,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True, lonmean=True)

    lont,wuaTa, cuaTa = composite_var(uaTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True, lonmean=True)

    #Integrate over the mixed layer
    iwucTa, icucTa = integrate_over_ml(wucTa, cucTa)
    iwuaTc, icuaTc = integrate_over_ml(wuaTc, cuaTc)
    iwuaTa, icuaTa = integrate_over_ml(wuaTa, cuaTa)

    del u_clim,u_anom,ucTa,uaTc,uaTa,wucTa,cucTa,wuaTc,cuaTc,wuaTa,cuaTa
    gc.collect()

    #Meridional advection
    dTady = diff_axis(np.array(pot_temp_anom),axis=-2)/(np.diff(lat)[0]*111320.)*\
    (1/3.80517e-7)
    dTcdy = diff_axis(np.array(pot_temp_clim),axis=-2)/(np.diff(lat)[0]*111320.)*\
    (1/3.80517e-7)

    vcTa = (-1.)*v_clim*dTady
    vaTc = (-1.)*v_anom*dTcdy
    vaTa = (-1.)*v_anom*dTady

    #Composite region
    lont, wvcTa, cvcTa = composite_var(vcTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True,lonmean=True)

    lont, wvaTc, cvaTc = composite_var(vaTc,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True, lonmean=True)

    lont,wvaTa, cvaTa = composite_var(vaTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True, lonmean=True)

    #Integrate over the mixed layer
    iwvcTa, icvcTa = integrate_over_ml(wvcTa, cvcTa)
    iwvaTc, icvaTc = integrate_over_ml(wvaTc, cvaTc)
    iwvaTa, icvaTa = integrate_over_ml(wvaTa, cvaTa)

    del v_clim,v_anom,vcTa,vaTc,vaTa,wvcTa,cvcTa,wvaTc,cvaTc,wvaTa,cvaTa
    gc.collect()

    #Vertical advection
    dTadz = diff_axis(np.array(pot_temp_anom),axis=-3)/(np.diff(lev)[0])*\
    (1/3.80517e-7)
    dTcdz = diff_axis(np.array(pot_temp_clim),axis=-3)/(np.diff(lev)[0])*\
    (1/3.80517e-7)

    wcTa = (-1.)*w_clim*dTadz
    waTc = (-1.)*w_anom*dTcdz
    waTa = (-1.)*w_anom*dTadz

    #Composite region
    lont, wwcTa, cwcTa = composite_var(wcTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True,lonmean=True)

    lont, wwaTc, cwaTc = composite_var(waTc,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True, lonmean=True)

    lont,wwaTa, cwaTa = composite_var(waTa,\
        iw_yr0,iw_yr1,ic_yr0,ic_yr1,lon,lat,\
        lev,lon1,lon2,lat1,lat2,lev1=lev1,lev2=lev2,\
        em=False,latmean=True, lonmean=True)

    #Integrate over the mixed layer
    iwwcTa, icwcTa = integrate_over_ml(wwcTa, cwcTa)
    iwwaTc, icwaTc = integrate_over_ml(wwaTc, cwaTc)
    iwwaTa, icwaTa = integrate_over_ml(wwaTa, cwaTa)

    del w_clim,w_anom,wcTa,waTc,waTa,wwcTa,cwcTa,wwaTc,cwaTc,wwaTa,cwaTa
    gc.collect()

    hb_warm = {'dt':iwdTa,'Q':wQ,
    'ucTa':iwucTa, 'uaTc':iwuaTc, 'uaTa':iwuaTa,\
    'vcTa':iwvcTa, 'vaTc':iwvaTc, 'vaTa':iwvaTa,\
    'wcTa':iwwcTa, 'waTc':iwwaTc, 'waTa':iwwaTa,\
    }
    hb_cold = {'dt':icdTa,'Q':cQ,
    'ucTa':icucTa, 'uaTc':icuaTc, 'uaTa':icuaTa,\
    'vcTa':icvcTa, 'vaTc':icvaTc, 'vaTa':icvaTa,\
    'wcTa':icwcTa, 'waTc':icwaTc, 'waTa':icwaTa,\
    }

    return hb_warm, hb_cold
#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ind') #atlnino, e_index, c_index
    parser.add_argument('dur') #first, last, all
    parser.add_argument('exp') #Mclm or Oclm
    parser.add_argument('nem') #0-20
    parser.add_argument('tem') #Total number of ensembles
    # parser.add_argument('filter') #True or False
    args = parser.parse_args()

    iind = str(args.ind)
    idur = str(args.dur)
    iexp = str(args.exp)
    inem = int(args.nem)
    item = int(args.tem)
    # ifilter=bool(args.filter)

    if iind=='atlnino':
        start=5
    else:
        start=10

    if iind=='atlnino':
        lon1 = -20%360
        lon2 = 360
        lat1 = -5
        lat2 = 5

        lev1 = 0
        if iexp=='obs':
            lev2 = 32.
        else:
            lev2 = 38.

    elif iind=='e_index':
        lon1 = -150%360
        lon2 = -90%360
        lat1 = -5
        lat2 = 5

        lev1 = 0.
        if iexp=='obs':
            lev2 = 50.
        else:
            lev2 = 45.

    elif iind=='c_index':
        lon1 = 160%360
        lon2 = -150%360
        lat1 = -5
        lat2 = 5

        lev1 = 0.
        if iexp=='obs':
            lev2 = 50.
        else:
            lev2 = 45.

    #Load index
    klepto_indices = klepto.archives.dir_archive('klepto_indices', serialized=True, cached=False)
    if iexp=='obs':
        ind = klepto_indices[iexp+'_'+iind]
    else:
        ind = klepto_indices[iexp+'_'+iind][inem]

    if iind=='atlnino':
        #Select and average season
        ind_s = np.concatenate([ind[i:i+4] for i in range(start,ind.shape[0]-12,12)],axis=0)
        #Mean of season
        ind_s_mean = np.nanmean(np.stack(np.split(ind_s, ind_s.shape[0]//4, axis=0), axis=0),axis=1)
        #Normalise the index
        ind_n = normalise_index(ind_s_mean)
    else:
        ind_n = ind

    #Indices of years with index more than x=1 (warm events)
    i_warm = count_warm_cold(ind_n)[2]

    #Indices of years with index less than x=-1 (cold events)
    i_cold = count_warm_cold(ind_n)[3]

    ##Load detrended variables(climatologies and anomalies)
    # if ifilter==False:
    m = 65
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    if iexp=='obs':
        #Temperature
        pot_temp_mean = klepto_atm_detrend_trop[iexp+'_pot_temp_mean'][:-12]
        pot_temp_anom = klepto_atm_detrend_trop[iexp+'_pot_temp_anom'][:-12]
        pot_temp_clim = klepto_atm_detrend_trop[iexp+'_pot_temp_clim']
        pot_temp_clim = np.tile(pot_temp_clim,(m,1,1,1))

        #Net surface heat flux
        tflux_mean = klepto_atm_detrend_trop[iexp+'_tflux_mean'][:-12]
        tflux_anom = klepto_atm_detrend_trop[iexp+'_tflux_anom'][:-12]

        #Zonal current
        u_mean = klepto_atm_detrend_trop[iexp+'_u_mean'][:-12]
        u_anom = klepto_atm_detrend_trop[iexp+'_u_anom'][:-12]
        u_clim = klepto_atm_detrend_trop[iexp+'_u_clim']
        u_clim = np.tile(u_clim,(m,1,1,1))

        #Meridional current
        v_mean = klepto_atm_detrend_trop[iexp+'_v_mean'][:-12]
        v_anom = klepto_atm_detrend_trop[iexp+'_v_anom'][:-12]
        v_clim = klepto_atm_detrend_trop[iexp+'_v_clim']
        v_clim = np.tile(v_clim,(m,1,1,1))

        #Vertical current
        w_mean = klepto_atm_detrend_trop[iexp+'_w_mean'][:-12]
        w_anom = klepto_atm_detrend_trop[iexp+'_w_anom'][:-12]
        w_clim = klepto_atm_detrend_trop[iexp+'_w_clim'][:-12]

        #lon lat lev time
        # time = klepto_atm_detrend_trop['time']
        lon = klepto_atm_detrend_trop['obs_lon']
        lat = klepto_atm_detrend_trop['obs_lat']
        lev = klepto_atm_detrend_trop['obs_lev']

    elif iexp=='cmip5':
        #temperature
        pot_temp_mean = klepto_atm_detrend_trop[iexp+'_pot_temp_mean'][inem][:-12]
        pot_temp_anom = klepto_atm_detrend_trop[iexp+'_pot_temp_anom'][inem][:-12]
        pot_temp_clim = klepto_atm_detrend_trop[iexp+'_pot_temp_clim'][inem]
        pot_temp_clim = np.tile(pot_temp_clim,(m,1,1,1))

        #Net surface heat flux
        tflux_mean = klepto_atm_detrend_trop[iexp+'_tflux_mean'][inem][:-12]
        tflux_anom = klepto_atm_detrend_trop[iexp+'_tflux_anom'][inem][:-12]

        #Zonal current
        u_mean = klepto_atm_detrend_trop[iexp+'_u_mean'][inem][:-12]
        u_anom = klepto_atm_detrend_trop[iexp+'_u_anom'][inem][:-12]
        u_clim = klepto_atm_detrend_trop[iexp+'_u_clim'][inem]
        u_clim = np.tile(u_clim,(m,1,1,1))

        #Meridional current
        v_mean = klepto_atm_detrend_trop[iexp+'_v_mean'][inem][:-12]
        v_anom = klepto_atm_detrend_trop[iexp+'_v_anom'][inem][:-12]
        v_clim = klepto_atm_detrend_trop[iexp+'_v_clim'][inem]
        v_clim = np.tile(v_clim,(m,1,1,1))

        #Vertical current
        w_mean = klepto_atm_detrend_trop[iexp+'_w_mean'][inem][:-12]
        w_anom = klepto_atm_detrend_trop[iexp+'_w_anom'][inem][:-12]
        w_clim = klepto_atm_detrend_trop[iexp+'_w_clim'][inem][:-12]

        #lon lat lev time
        # time = klepto_atm_detrend_trop['time']
        lon = klepto_atm_detrend_trop['lon']
        lat = klepto_atm_detrend_trop['lat']
        lev = klepto_atm_detrend_trop['lev']

    elif (iexp=='Mclm') or (iexp=='Oclm'):
        #temperature
        pot_temp_mean = klepto_atm_detrend_trop[iexp+'_pot_temp_mean'][inem][:-12]
        pot_temp_anom = klepto_atm_detrend_trop[iexp+'_pot_temp_anom'][inem][:-12]
        pot_temp_clim = klepto_atm_detrend_trop[iexp+'_pot_temp_clim'][inem]
        pot_temp_clim = np.tile(pot_temp_clim,(m,1,1,1))

        #Net surface heat flux
        tflux_mean = klepto_atm_detrend_trop[iexp+'_tflux_mean'][inem][:-12]
        tflux_anom = klepto_atm_detrend_trop[iexp+'_tflux_anom'][inem][:-12]

        #Zonal current
        u_mean = klepto_atm_detrend_trop[iexp+'_u_mean'][inem][:-12]
        u_anom = klepto_atm_detrend_trop[iexp+'_u_anom'][inem][:-12]
        u_clim = klepto_atm_detrend_trop[iexp+'_u_clim'][inem]
        u_clim = np.tile(u_clim,(m,1,1,1))

        #Meridional current
        v_mean = klepto_atm_detrend_trop[iexp+'_v_mean'][inem][:-12]
        v_anom = klepto_atm_detrend_trop[iexp+'_v_anom'][inem][:-12]
        v_clim = klepto_atm_detrend_trop[iexp+'_v_clim'][inem]
        v_clim = np.tile(v_clim,(m,1,1,1))

        #Vertical current
        wm_ctl1 = klepto_atm_detrend_trop[iexp+'_ctl1_w_mean']
        wm_ctl2 = klepto_atm_detrend_trop[iexp+'_ctl2_w_mean']
        wm_io1 = klepto_atm_detrend_trop[iexp+'_io1_w_mean']
        wm_io2 = klepto_atm_detrend_trop[iexp+'_io2_w_mean']
        w_mean = np.concatenate((wm_ctl1,wm_ctl2,wm_io1,wm_io2),axis=0)[inem][:-12]

        del wm_ctl1, wm_ctl2, wm_io1, wm_io2
        gc.collect()

        wa_ctl1 = klepto_atm_detrend_trop[iexp+'_ctl1_w_anom']
        wa_ctl2 = klepto_atm_detrend_trop[iexp+'_ctl2_w_anom']
        wa_io1 = klepto_atm_detrend_trop[iexp+'_io1_w_anom']
        wa_io2 = klepto_atm_detrend_trop[iexp+'_io2_w_anom']
        w_anom = np.concatenate((wa_ctl1,wa_ctl2,wa_io1,wa_io2),axis=0)[inem][:-12]

        del wa_ctl1, wa_ctl2, wa_io1, wa_io2
        gc.collect()

        wc_ctl1 = klepto_atm_detrend_trop[iexp+'_ctl1_w_clim']
        wc_ctl2 = klepto_atm_detrend_trop[iexp+'_ctl2_w_clim']
        wc_io1 = klepto_atm_detrend_trop[iexp+'_io1_w_clim']
        wc_io2 = klepto_atm_detrend_trop[iexp+'_io2_w_clim']
        w_clim = np.concatenate((wc_ctl1,wc_ctl2,wc_io1,wc_io2),axis=0)[inem][:-12]

        del wc_ctl1, wc_ctl2, wc_io1, wc_io2
        gc.collect()

        #lon lat lev time
        # time = klepto_atm_detrend_trop['time']
        lon = klepto_atm_detrend_trop['lon']
        lat = klepto_atm_detrend_trop['lat']
        lev = klepto_atm_detrend_trop['lev']

    # elif ifilter==True:
    #     klepto_atm_detrend_trop_flt = klepto.archives.dir_archive('klepto_atm_detrend_trop_flt',\
    #     serialized=True, cached=False)
    #     if iexp=='obs':
    #         #Temperature
    #         pot_temp_mean = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_mean'][:-12]
    #         pot_temp_anom = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_anom'][:-12]
    #         pot_temp_clim = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_clim']
    #
    #         #Net surface heat flux
    #         tflux_mean = klepto_atm_detrend_trop_flt[iexp+'_tflux_mean'][:-12]
    #         tflux_anom = klepto_atm_detrend_trop_flt[iexp+'_tflux_anom'][:-12]
    #
    #         #Zonal current
    #         u_mean = klepto_atm_detrend_trop_flt[iexp+'_u_mean'][:-12]
    #         u_anom = klepto_atm_detrend_trop_flt[iexp+'_u_anom'][:-12]
    #         u_clim = klepto_atm_detrend_trop_flt[iexp+'_u_clim']
    #
    #         #Meridional current
    #         v_mean = klepto_atm_detrend_trop_flt[iexp+'_v_mean'][:-12]
    #         v_anom = klepto_atm_detrend_trop_flt[iexp+'_v_anom'][:-12]
    #         v_clim = klepto_atm_detrend_trop_flt[iexp+'_v_clim']
    #
    #         #Vertical current
    #         w_mean = klepto_atm_detrend_trop_flt[iexp+'_w_mean'][:-12]
    #         w_anom = klepto_atm_detrend_trop_flt[iexp+'_w_anom'][:-12]
    #         w_clim = klepto_atm_detrend_trop_flt[iexp+'_w_clim']
    #
    #         #lon lat lev time
    #         # time = klepto_atm_detrend_trop['time']
    #         lon = klepto_atm_detrend_trop_flt['obs_lon']
    #         lat = klepto_atm_detrend_trop_flt['obs_lat']
    #         lev = klepto_atm_detrend_trop_flt['obs_lev']
    #
    #     elif iexp=='cmip5':
    #         #Temperature
    #         pot_temp_mean = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_mean'][inem][:-12]
    #         pot_temp_anom = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_anom'][inem][:-12]
    #         pot_temp_clim = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_clim'][inem]
    #
    #         #Net surface heat flux
    #         tflux_mean = klepto_atm_detrend_trop_flt[iexp+'_tflux_mean'][inem][:-12]
    #         tflux_anom = klepto_atm_detrend_trop_flt[iexp+'_tflux_anom'][inem][:-12]
    #
    #         #Zonal current
    #         u_mean = klepto_atm_detrend_trop_flt[iexp+'_u_mean'][inem][:-12]
    #         u_anom = klepto_atm_detrend_trop_flt[iexp+'_u_anom'][inem][:-12]
    #         u_clim = klepto_atm_detrend_trop_flt[iexp+'_u_clim'][inem]
    #
    #         #Meridional current
    #         v_mean = klepto_atm_detrend_trop_flt[iexp+'_v_mean'][inem][:-12]
    #         v_anom = klepto_atm_detrend_trop_flt[iexp+'_v_anom'][inem][:-12]
    #         v_clim = klepto_atm_detrend_trop_flt[iexp+'_v_clim'][inem]
    #
    #         #Vertical current
    #         w_mean = klepto_atm_detrend_trop_flt[iexp+'_w_mean'][inem][:-12]
    #         w_anom = klepto_atm_detrend_trop_flt[iexp+'_w_anom'][inem][:-12]
    #         w_clim = klepto_atm_detrend_trop_flt[iexp+'_w_clim'][inem]
    #
    #         #lon lat lev time
    #         # time = klepto_atm_detrend_trop['time']
    #         lon = klepto_atm_detrend_trop_flt['lon']
    #         lat = klepto_atm_detrend_trop_flt['lat']
    #         lev = klepto_atm_detrend_trop_flt['lev']
    #
    #     elif iexp=='Mclm':
    #         #Temperature
    #         pot_tempm_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_pot_temp_mean']
    #         pot_tempm_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_pot_temp_mean']
    #         pot_tempm_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_pot_temp_mean']
    #         pot_tempm_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_pot_temp_mean']
    #         pot_temp_mean = np.concatenate((pot_tempm_ctl1,pot_tempm_ctl2,pot_tempm_io1,pot_tempm_io2),axis=0)[inem][:-12]
    #
    #         del pot_tempm_ctl1, pot_tempm_ctl2, pot_tempm_io1, pot_tempm_io2
    #         gc.collect()
    #
    #         pot_tempa_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_pot_temp_anom']
    #         pot_tempa_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_pot_temp_anom']
    #         pot_tempa_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_pot_temp_anom']
    #         pot_tempa_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_pot_temp_anom']
    #         pot_temp_anom = np.concatenate((pot_tempa_ctl1,pot_tempa_ctl2,pot_tempa_io1,pot_tempa_io2),axis=0)[inem][:-12]
    #
    #         del pot_tempa_ctl1, pot_tempa_ctl2, pot_tempa_io1, pot_tempa_io2
    #         gc.collect()
    #
    #         pot_tempc_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_pot_temp_clim']
    #         pot_tempc_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_pot_temp_clim']
    #         pot_tempc_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_pot_temp_clim']
    #         pot_tempc_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_pot_temp_clim']
    #         pot_temp_clim = np.concatenate((pot_tempc_ctl1,pot_tempc_ctl2,pot_tempc_io1,pot_tempc_io2),axis=0)[inem][:-12]
    #
    #         del pot_tempc_ctl1, pot_tempc_ctl2, pot_tempc_io1, pot_tempc_io2
    #         gc.collect()
    #
    #         #Net surface heat flux
    #         tflux_mean = klepto_atm_detrend_trop_flt[iexp+'_tflux_mean'][inem][:-12]
    #         tflux_anom = klepto_atm_detrend_trop_flt[iexp+'_tflux_anom'][inem][:-12]
    #
    #         #Zonal current
    #         um_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_u_mean']
    #         um_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_u_mean']
    #         um_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_u_mean']
    #         um_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_u_mean']
    #         u_mean = np.concatenate((um_ctl1,um_ctl2,um_io1,um_io2),axis=0)[inem][:-12]
    #
    #         del um_ctl1, um_ctl2, um_io1, um_io2
    #         gc.collect()
    #
    #         ua_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_u_anom']
    #         ua_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_u_anom']
    #         ua_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_u_anom']
    #         ua_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_u_anom']
    #         u_anom = np.concatenate((ua_ctl1,ua_ctl2,ua_io1,ua_io2),axis=0)[inem][:-12]
    #
    #         del ua_ctl1, ua_ctl2, ua_io1, ua_io2
    #         gc.collect()
    #
    #         uc_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_u_clim']
    #         uc_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_u_clim']
    #         uc_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_u_clim']
    #         uc_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_u_clim']
    #         u_clim = np.concatenate((uc_ctl1,uc_ctl2,uc_io1,uc_io2),axis=0)[inem][:-12]
    #
    #         del uc_ctl1, uc_ctl2, uc_io1, uc_io2
    #         gc.collect()
    #
    #         #Meridional current
    #         vm_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_v_mean']
    #         vm_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_v_mean']
    #         vm_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_v_mean']
    #         vm_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_v_mean']
    #         v_mean = np.concatenate((vm_ctl1,vm_ctl2,vm_io1,vm_io2),axis=0)[inem][:-12]
    #
    #         del vm_ctl1, vm_ctl2, vm_io1, vm_io2
    #         gc.collect()
    #
    #         va_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_v_anom']
    #         va_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_v_anom']
    #         va_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_v_anom']
    #         va_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_v_anom']
    #         v_anom = np.concatenate((va_ctl1,va_ctl2,va_io1,va_io2),axis=0)[inem][:-12]
    #
    #         del va_ctl1, va_ctl2, va_io1, va_io2
    #         gc.collect()
    #
    #         vc_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_v_clim']
    #         vc_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_v_clim']
    #         vc_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_v_clim']
    #         vc_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_v_clim']
    #         v_clim = np.concatenate((vc_ctl1,vc_ctl2,vc_io1,vc_io2),axis=0)[inem][:-12]
    #
    #         del vc_ctl1, vc_ctl2, vc_io1, vc_io2
    #         gc.collect()
    #
    #         #Vertical current
    #         wm_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_w_mean']
    #         wm_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_w_mean']
    #         wm_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_w_mean']
    #         wm_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_w_mean']
    #         w_mean = np.concatenate((wm_ctl1,wm_ctl2,wm_io1,wm_io2),axis=0)[inem][:-12]
    #
    #         del wm_ctl1, wm_ctl2, wm_io1, wm_io2
    #         gc.collect()
    #
    #         wa_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_w_anom']
    #         wa_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_w_anom']
    #         wa_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_w_anom']
    #         wa_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_w_anom']
    #         w_anom = np.concatenate((wa_ctl1,wa_ctl2,wa_io1,wa_io2),axis=0)[inem][:-12]
    #
    #         del wa_ctl1, wa_ctl2, wa_io1, wa_io2
    #         gc.collect()
    #
    #         wc_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_w_clim']
    #         wc_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_w_clim']
    #         wc_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_w_clim']
    #         wc_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_w_clim']
    #         w_clim = np.concatenate((wc_ctl1,wc_ctl2,wc_io1,wc_io2),axis=0)[inem][:-12]
    #
    #         del wc_ctl1, wc_ctl2, wc_io1, wc_io2
    #         gc.collect()
    #
    #         #lon lat lev time
    #         # time = klepto_atm_detrend_trop['time']
    #         lon = klepto_atm_detrend_trop_flt['lon']
    #         lat = klepto_atm_detrend_trop_flt['lat']
    #         lev = klepto_atm_detrend_trop_flt['lev']
    #Warm events
    # iw_yr_1 = np.array(i_warm)-1
    iw_yr0 = np.array(i_warm)
    iw_yr1 = np.array(i_warm)+1

    #Cold events
    # ic_yr_1 = np.array(i_cold)-1
    ic_yr0 = np.array(i_cold)
    ic_yr1 = np.array(i_cold)+1

    #Heat budget
    if idur=='first':
        im = pot_temp_anom.shape[0]//2
        iw_yr0 = np.array([i for i in iw_yr0 if i <= im//12])
        iw_yr1 = np.array([i for i in iw_yr1 if i <= (im//12)+1])
        ic_yr0 = np.array([i for i in ic_yr0 if i <= im//12])
        ic_yr1 = np.array([i for i in ic_yr1 if i <= (im//12)+1])
        hb_warm, hb_cold = heat_budget(pot_temp_mean, pot_temp_anom, \
        pot_temp_clim, tflux_anom,
        u_clim, u_anom, v_clim, v_anom, w_clim, w_anom,
        iw_yr0, iw_yr1, ic_yr0, ic_yr1,
        lon, lat, lev,
        lon1, lon2, lat1, lat2, lev1, lev2, m)
    elif idur=='last':
        im = pot_temp_anom.shape[0]//2
        iw_yr0 = np.array([i for i in iw_yr0 if i >= im//12])
        iw_yr1 = np.array([i for i in iw_yr1 if i >= (im//12)+1])
        ic_yr0 = np.array([i for i in ic_yr0 if i >= im//12])
        ic_yr1 = np.array([i for i in ic_yr1 if i >= (im//12)+1])
        hb_warm, hb_cold = heat_budget(pot_temp_mean, pot_temp_anom, \
        pot_temp_clim, tflux_anom,
        u_clim, u_anom, v_clim, v_anom, w_clim, w_anom,
        iw_yr0, iw_yr1, ic_yr0, ic_yr1,
        lon, lat, lev,
        lon1, lon2, lat1, lat2, lev1, lev2, m)
    else:
        im = pot_temp_anom.shape[0]
        hb_warm, hb_cold = heat_budget(pot_temp_mean, pot_temp_anom, pot_temp_clim, tflux_anom,
        u_clim, u_anom, v_clim, v_anom, w_clim, w_anom,
        iw_yr0, iw_yr1, ic_yr0, ic_yr1,
        lon, lat, lev,
        lon1, lon2, lat1, lat2, lev1, lev2, m)

    #Save to klepto
    klepto_hba = klepto.archives.dir_archive('klepto_hba',\
    serialized=True, cached=False)
    if inem<=item-1:
        klepto_hba[iind+'_warm_'+str(iexp)+'_'+str(idur)+'_'+str(inem+1).zfill(2)] = hb_warm
        klepto_hba[iind+'_cold_'+str(iexp)+'_'+str(idur)+'_'+str(inem+1).zfill(2)] = hb_cold
    elif inem>=item:
        klepto_hba[iind+'_warm_'+str(iexp)+'T_'+str(idur)+'_'+str(inem-item+1).zfill(2)] = hb_warm
        klepto_hba[iind+'_cold_'+str(iexp)+'T_'+str(idur)+'_'+str(inem-item+1).zfill(2)] = hb_cold
