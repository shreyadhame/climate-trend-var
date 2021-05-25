
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Benguela Nino index"
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

def calc_bengnino_index(ssta, lon, lat, em=True):
    lat1 = -25
    lat2 = -15
    lon1  = 360+5
    lon2 = 360+15
    lev=[]

    lons, lats, levs, ssta_ta = selreg(ssta,lon,lat,lev,lon1,lon2,lat1,lat2,lev1=-999.,lev2=-999.,em=em)

    bengnino = wgtvar(ma.masked_invalid(ssta_ta),ma.masked_invalid(lons),\
    ma.masked_invalid(lats),lev,em)

    return bengnino.squeeze()

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
        em=False
    else:
        lon = klepto_atm_detrend_trop['lon']
        lat = klepto_atm_detrend_trop['lat']
        em=True

    #Calculate Benguela Nino index
    bengnino = calc_bengnino_index(ssta,lon,lat,em=em)

    #Save to klepto
    klepto_indices = klepto.archives.dir_archive('klepto_indices',\
    serialized=True, cached=False)
    klepto_indices[iexp+'_bengnino']=bengnino
