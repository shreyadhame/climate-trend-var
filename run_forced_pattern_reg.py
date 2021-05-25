import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')
import gc
import klepto
import numpy.ma as ma
from lfca_fpf import *

import matplotlib.pyplot as plt
from clim_indices import clim_anom
#=================================================================
seas
multivar

#Load SST, SLP and precip anomalies
klepto_atm_data_xr = klepto.archives.dir_archive('klepto_atm_data_xr',\
serialized=True, cached=False)
sst=klepto_atm_data_xr['sst']
slp=klepto_atm_data_xr['slp']
pr=klepto_atm_data_xr['pr']
time=klepto_atm_data_xr['time']
lat=klepto_atm_data_xr['lat']
lon=klepto_atm_data_xr['lon']

#Months for seas selection
if seas == 'DJF':
    smonths = [12,1,2]
elif seas == 'MAM':
    smonths = [3,4,5]
elif seas == 'JJA':
    smonths = [6,7,8]
elif seas == 'SON':
    smonths = [9,10,11]
else:
    pass

#Select season
if seas != 'AM':
    sst = [season(sst[i],months=smonths) for i in range(len(sst))]
    slp = [season(slp[i],months=smonths) for i in range(len(slp))]
    pr = [season(pr[i],months=smonths) for i in range(len(pr))]
    time = season(time,months=smonths)
else:
    pass

#Stack and roll
sst = np.stack(sst,axis=0); sstr = np.roll(sst,80,axis=-1)
#Calculate anomalies
ssta = clim_anom(sstr,time,start=None,nmonths=12,nem=12)[1]

if multivar == True:
    #Normalise by trace of covariance matrix

    #Concatenate matrices

else:





#Separate experiment and control | Replace masked values with zero
nem = 6
ssta_ctl = np.nan_to_num(ssta[:nem])
ssta_io = np.nan_to_num(ssta[nem:])

#reshape data
ssta_ctl_em = ssta_ctl.mean(axis=0)
ssta_io_em = ssta_io.mean(axis=0)
ssta_ctl_all = np.concatenate(ssta_ctl)
ssta_io_all = np.concatenate(ssta_io)
ssta_ctl_all = np.concatenate(ssta_ctl)

areawt = np.cos(np.tile(abs(lat.values[:,None])*np.pi/180,(1,len(lon))))
domain = np.ones(np.shape(areawt))
x,y = np.meshgrid(lon.values,lat.values)

#Eastern Equatorial Pacific region
domain[np.where(y<-10)] = 0.
domain[np.where(y>10)] = 0.
domain[np.where(x<-150%360)] = 0.
domain[np.where(x>-80%360)] = 0.

#Western Equatorial Pacific region
domain[np.where(y<-10)] = 0.
domain[np.where(y>10)] = 0.
domain[np.where(x<140%360)] = 0.
domain[np.where(x>-150%360)] = 0.

# #North Equatorial Atlantic region
# domain[np.where(y<0)] = 0.
# domain[np.where(y>15)] = 0.
# domain[np.where(x<-70%360)] = 0.
# domain[np.where(x>360)] = 0.

# #South Equatorial Atlantic region
# domain[np.where(y<-15)] = 0.
# domain[np.where(y>0)] = 0.
# domain[np.where(x<-50%360)] = 0.
# domain[np.where(x>360)] = 0.

# #North Atlantic region
# domain[np.where(y<40)] = 0.
# domain[np.where(y>60)] = 0.
# domain[np.where(x<-80%360)] = 0.
# domain[np.where(x>360)] = 0.
#
# #North Pacific region
# domain[np.where(y<10)] = 0.
# domain[np.where(y>60)] = 0.
# domain[np.where(x<110)] = 0.
# domain[np.where(x>-100%360)] = 0.

X_io = np.reshape(ssta_io_all,(ssta_io_all.shape[0],ssta_io_all.shape[1]*ssta_io_all.shape[2]))
Xe_io = np.reshape(ssta_io_em,(ssta_io_em.shape[0],ssta_io_em.shape[1]*ssta_io_em.shape[2]))
X_ctl = np.reshape(ssta_ctl_all,(ssta_ctl_all.shape[0],ssta_ctl_all.shape[1]*ssta_ctl_all.shape[2]))
Xe_ctl = np.reshape(ssta_ctl_em,(ssta_ctl_em.shape[0],ssta_ctl_em.shape[1]*ssta_ctl_em.shape[2]))
area_weights = np.reshape(areawt,(1,areawt.shape[0]*areawt.shape[1]))
domain = np.reshape(domain,(1,domain.shape[0]*domain.shape[1]))

icol_ret = np.where((area_weights!=0) & (domain!=0))
icol_disc = np.where((area_weights==0) | (domain==0))
X_io = X_io[:,icol_ret[1]]
Xe_io = Xe_io[:,icol_ret[1]]
X_ctl = X_ctl[:,icol_ret[1]]
Xe_ctl = Xe_ctl[:,icol_ret[1]]
area_weights = area_weights[:,icol_ret[1]]

#scale by square root of grid cell area such that covariance is area
normvec = area_weights.T/np.sum(area_weights)
scale = np.sqrt(normvec)

#Parameters
truncation = 200  #number of EOFs
M = 12 #number of Forced Patterns to retain in FP filtering

# Calculate forced patterns / Signal-to-noise maximizing EOF analysis (IOtrend and control)

tk_io, FPs_io, fingerprints_io, s_io, pvar_io, pcs_io, EOFs_io, N_io, pvar_FPs_io,\
s_eofs_io = forced_pattern_analysis(X_io, Xe_io, truncation, scale)

klepto_fpa_sst = klepto.archives.dir_archive('klepto_fpa_sea_sst',serialized=True, cached=False)
klepto_fpa_sst['tk_io'] = tk_io
klepto_fpa_sst['FPs_io'] = FPs_io
klepto_fpa_sst['fingerprints_io'] = fingerprints_io
klepto_fpa_sst['s_io'] = s_io
klepto_fpa_sst['pvar_io'] = pvar_io
klepto_fpa_sst['pcs_io'] = pcs_io
klepto_fpa_sst['EOFs_io'] = EOFs_io
klepto_fpa_sst['N_io'] = N_io
klepto_fpa_sst['pvar_FPs_io'] = pvar_FPs_io
klepto_fpa_sst['s_eofs_io'] = s_eofs_io

tk_ctl, FPs_ctl, fingerprints_ctl, s_ctl, pvar_ctl, pcs_ctl, EOFs_ctl, N_ctl, pvar_FPs_ctl,\
s_eofs_ctl = forced_pattern_analysis(X_ctl, Xe_ctl, truncation, scale)

klepto_fpa_sst = klepto.archives.dir_archive('klepto_fpa_sea_sst',serialized=True, cached=False)
klepto_fpa_sst['tk_ctl'] = tk_ctl
klepto_fpa_sst['FPs_ctl'] = FPs_ctl
klepto_fpa_sst['fingerprints_ctl'] = fingerprints_ctl
klepto_fpa_sst['s_ctl'] = s_ctl
klepto_fpa_sst['pvar_ctl'] = pvar_ctl
klepto_fpa_sst['pcs_ctl'] = pcs_ctl
klepto_fpa_sst['EOFs_ctl'] = EOFs_ctl
klepto_fpa_sst['N_ctl'] = N_ctl
klepto_fpa_sst['pvar_FPs_ctl'] = pvar_FPs_ctl
klepto_fpa_sst['s_eofs_ctl'] = s_eofs_ctl
