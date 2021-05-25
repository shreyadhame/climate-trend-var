
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Atlantic Nino index"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
#General modules
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts')

import argparse
import klepto
import numpy as np
import numpy.ma as ma

#My modules
from read import selreg, wgtvar
#=================================================================

def calc_atlnino_index(ssta, lon, lat, lat1, lat2, em=True):
    lon1  = -20%360
    lon2 = 360
    lev=[]

    # lon_ta=lon.sel(longitude=slice(lon1,lon2))
    # lat_ta=lat.sel(latitude=slice(lat1,lat2))

    lons, lats, levs, ssta_ta = selreg(ssta,lon,lat,lev,lon1,lon2,lat1,lat2,lev1=-999.,lev2=-999.,em=em)

    # atlnino = ssta_ta.mean(axis=-1).mean(axis=-1)
    atlnino = wgtvar(ma.masked_invalid(ssta_ta),ma.masked_invalid(lons),\
    ma.masked_invalid(lats),lev,em)

    return atlnino.squeeze()

#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp') #Mclm, Oclm, cmip5, obs
    args = parser.parse_args()

    iexp = str(args.exp)

    #Load detrended tropical SST anomalies from klepto
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    ssta = klepto_atm_detrend_trop[iexp+'_sst_anom']
    if iexp=='obs':
        lon = klepto_atm_detrend_trop['obs_lon']
        lat = klepto_atm_detrend_trop['obs_lat']
    else:
        lon = klepto_atm_detrend_trop['lon']
        lat = klepto_atm_detrend_trop['lat']

    if (iexp=='Mclm') or (iexp=='Oclm') or (iexp=='cmip5'):
        lat1=-5
        lat2=5
        #Calculate Atl Nino index
        atlnino = calc_atlnino_index(ssta,lon,lat,lat1,lat2,em=True)
    elif (iexp=='obs'):
        lat1=-5
        lat2=5
        #Calculate Atl Nino index
        atlnino = calc_atlnino_index(ssta,lon,lat,lat1,lat2,em=False)

    #Save to klepto
    klepto_indices = klepto.archives.dir_archive('klepto_indices',\
    serialized=True, cached=False)
    klepto_indices[iexp+'_atlnino']=atlnino
