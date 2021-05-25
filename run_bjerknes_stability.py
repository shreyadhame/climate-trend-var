#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Bjerknes Stability analysis"
__reference__ = "Kim et al. (2011), Lübbecke and McPhaden (2013), \
https://www.nhc.noaa.gov/gccalc.shtml"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#=================================================================
import os
import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')

import argparse
import bottleneck as bn
import gc
import itertools
import klepto
import numpy as np
import numpy.ma as ma
from scipy import stats
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import warnings
warnings.filterwarnings("ignore")

from read import selreg
from climanom import movingaverage, sel_season

#=================================================================
def diff_axis(var,axis=-1):
    vard = np.diff(var,axis=axis)
    if vard.ndim==1:
        vard = np.insert(vard,-1,[vard[-1]],axis=0)
    elif vard.ndim==2:
        if axis==-1:
            vard = np.insert(vard,-1,[vard[:,-1]],axis=axis)
        elif axis==-2:
            vard = np.insert(vard,-1,[vard[-1,:]],axis=axis)
    elif vard.ndim==3:
        if axis==-1:
            vard = np.insert(vard,-1,[vard[:,:,-1]],axis=axis)
        elif axis==-2:
            vard = np.insert(vard,-1,[vard[:,-1,:]],axis=axis)
        elif axis==-3:
            vard = np.insert(vard,-1,[vard[-1,:,:]],axis=axis)
    elif vard.ndim==4:
        if axis==-1:
            vard = np.insert(vard,-1,[vard[:,:,:,-1]],axis=axis)
        elif axis==-2:
            vard = np.insert(vard,-1,[vard[:,:,-1,:]],axis=axis)
        elif axis==-3:
            vard = np.insert(vard,-1,[vard[:,-1,:,:]],axis=axis)
        elif axis==-4:
            vard = np.insert(vard,-1,[vard[-1,:,:,:]],axis=axis)
    return vard

def outlierCleaner(predictions, x, y, pc):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (x, y, error).
    """
    tuples = []
    length = len(x)
    for i in range(length):
        tuples.append((x[i], y[i], y[i] - predictions[i]))

    differences_tuples = []
    for i in range(length):
        differences_tuples.append((abs(y[i] - predictions[i]), i))
    differences_sorted = sorted(differences_tuples)

    # Return the indices of the datapoints to be removed
    indices_to_remove = []
    for i in range(int(length/pc)):
        indices_to_remove.append(differences_sorted[length - 1 - i][1])
    indices_to_remove = sorted(indices_to_remove, reverse=True)

    # Remove the relevant tuples
    for i in indices_to_remove:
        del tuples[i]

    xn = np.array([tuples[i][0] for i in range(len(tuples))])
    yn = np.array([tuples[i][1] for i in range(len(tuples))])
    error = np.array([tuples[i][2] for i in range(len(tuples))])
    return xn, yn, error

def calc_lr(x1,x2,y,type='s'):
    #fit linear regression
    if type=='s':
        x = x1
#         lr = TheilSenRegressor(random_state=0)
        lr = LinearRegression()
        lr.fit(x.reshape(-1,1),y.reshape(-1,1))
#         coef = lr.coef_[0] #ts
        coef = list(itertools.chain(*lr.coef_))[0] #lr
    elif type=='m':
        lr = LinearRegression()
#         lr = TheilSenRegressor(random_state=0)
        x = np.stack((x1, x2), axis=-1)
        lr.fit(x,y)
        coef = lr.coef_
    return coef

def calc_lr_clean(x,y,pc):
    lr = LinearRegression()
    lr.fit(x.reshape(-1,1),y.reshape(-1,1))
    coef = list(itertools.chain(*lr.coef_))[0] #lr
    intercept = lr.intercept_
    predictions = coef*x + intercept

    #Remove 10% of outliers
    xn, yn, error = outlierCleaner(predictions, x, y,pc)
    #Calculate coefficient again
    lr = TheilSenRegressor(random_state=0)
    lr.fit(xn.reshape(-1,1),yn.reshape(-1,1))
    coef = lr.coef_[0] #ts
    return coef

def calc_box_av(var_box, meantype='b'):
    """
    Area/volume average over the box/boundaries
    """
    if meantype=='z':
        var_box_av = ((var_box[:,:,0] + var_box[:,:,-1])/2.).mean()
    elif meantype=='m':
        var_box_av = ((var_box[:,0,:] + var_box[:,-1,:])/2.).mean()
    elif meantype=='b':
        if var_box.ndim==3:
#             var_box_av = np.squeeze(np.apply_over_axes(np.mean,var_box,(1,2)))
            var_box_av = np.nanmean(np.nanmean(var_box,axis=-1),axis=-1)
        elif var_box.ndim==4:
#             var_box_av = np/.squeeze(np.apply_over_axes(np.mean,var_box,(1,2,3)))
            var_box_av = np.nanmean(np.nanmean(np.nanmean(var_box,axis=-1),axis=-1),axis=-1)
    return var_box_av

def convert_freq(var, mld, rho, shc=3850.): #3850J/kgC
    var_fr = var/(mld*rho*shc)
    return var_fr

#=================================================================
def bjerknes_index(ithetaa,itfluxa,itau_xa,\
    ium,iua,ivm,iwm_b,iwm,iwa_b,\
    ithetaa_wu,ithetaa_eu,ithetam,ithetaa_b,
    ilat,ilev,\
    ih1,\
    idx,idz,\
    ilx,ily,\
    irho):

    #Temp anomaly in eastern region
    thetaae =  calc_box_av(ithetaa)
    #Heat flux anomaly in eastern region
    qae = calc_box_av(itfluxa)
    #Calculate thermodynamic damping
    alphas = calc_lr(thetaae, [], qae)
    alphas_fr = convert_freq(alphas, ih1, irho)/3.17098e-8

    #Mean zonal currents in eastern region
    ume = np.nanmean(ium)
    u = ume/ilx

    #Mean meridional currents in the eastern region
    yvm = (-2.)*(ivm*(abs(ilat[np.newaxis, np.newaxis, :, np.newaxis])*111320.))
    yvme = np.nanmean(yvm)
    v = yvme/(ily**2)

    #Mean vertical currents in the eastern region (at the base of the mixed layer)
    wme = np.nanmean(iwm_b)
    #Calculate depth of maximum vertical velocity (Hm) in eastern region
    wmeu = [np.nanmean(iwm[:,i,:,:]) for i in range(iwm.shape[1])]
    wmax = np.where(wmeu==np.amax(wmeu))
    hm = float(ilev[wmax])
    print('Hm is '+str(hm))
    w = wme/hm

    #Calculate dynamical damping of mean currents
    ddmc_fr = ((u/3.17098e-8)+(v/3.17098e-8)+(w/3.17098e-8))*(-1.)

    #Zonal wind stress anomaly in tropical region
    tauxat = calc_box_av(itau_xa)
    #Calculate zonal wind response to SST
    mew = calc_lr(thetaae, [], tauxat)

    #Ocean heat content anomaly in the western region
    haw = calc_box_av(ithetaa_wu)
    #Ocean heat content anomaly in the eastern region
    hae = calc_box_av(ithetaa_eu)
    #Rate of discharge of heat content anomaly in the western region
    dhwdt = diff_axis(haw,axis=0)
    #Calculate thermocline slope response to zonal wind stress
    betah = calc_lr(tauxat, [], (hae - haw))

    #Zonal current anomaly in the eastern region
    uae = calc_box_av(iua)
    betau, betauh = calc_lr(tauxat, haw, uae, type='m')

    #Step function using mean upwelling at the base of the mixed layer in the eastern box
    H_w = np.zeros(iwm_b.shape)
    H_w[np.where(iwm_b > 0.)] = 1.
    #hwwa
    hwwa = ma.masked_where(H_w==0.,iwa_b)
    hwwae = calc_box_av(hwwa)
    #Calculate upwelling response to wind stress

    betaw = calc_lr(tauxat*(-1.), [], hwwae)
    if betaw < 0.:
        pc = 2
        while betaw < 0.:
            pc += 1
            betaw = calc_lr_clean(tauxat*(-1.), hwwae, pc=pc)
        print(pc)
    else:
        pass

    #hwtsub_ebbox
    hwtsub = ma.masked_where(H_w==0.,ithetaa_b)
    hwtsube = calc_box_av(hwtsub)

    #Calculate subsurface temperature anomalies response to thermocline depth
    ah = calc_lr(hae, [], hwtsube)

    #Mean zonal T gradient (dT/dx)
    dtx = diff_axis(ithetam,axis=-1) #time,lev,lat,lon
    dtdx = dtx/idx[np.newaxis,np.newaxis,np.newaxis,:]
    dtdxe = (-1.)*np.nanmean(dtdx)

    #Mean vertical T gradient (dT/dz)
    dtz = diff_axis(ithetam,axis=-3) #time,lev,lat,lon
    dtdz = dtz/idz[np.newaxis,:,np.newaxis,np.newaxis]
    dtdze = (-1.)*np.nanmean(dtdz)

    #hwwm
    hwwm = ma.masked_where(H_w==0.,iwm_b)
    hwwmhmah = (hwwm/hm)*ah
    hwwmhmahe = np.nanmean(hwwmhmah)

    #Calculate zonal advective feedback
    zaf = mew*betau*dtdxe
    zaf_fr = zaf/3.17098e-8

    #Calculate ekman feedback
    ekf = mew*betaw*dtdze
    ekf_fr = ekf/3.17098e-8

    #Calculate thermocline feedback
    thf = mew*betah*hwwmhmahe
    thf_fr = thf/3.17098e-8

    #Calculate bjerknes index
    bj = ddmc_fr + alphas_fr + ekf_fr + thf_fr + zaf_fr

#     def calc_f():
#         f = (betauh*(-1)*dtdxe)+(whme*ah)
#         return f

#     def calc_epsilon_ftilda():
#         epsilon,ftilda=calc_lr((-1.)*haw,(-1.)*tauxat,\
#         dhwdt,type='m')
#         return epsilon,ftilda

#     epsilon = calc_epsilon_ftilda()[0]
#     ftilda = calc_epsilon_ftilda()[1]

#     #w/h1
#     wmh1 = wm_ebbox/h1
#     wmh1e = bn.nanmean(wmh1)

#     Calculate frequency of the interannual oscillation
#     f = calc_f()
#     f_fr = f/3.17098e-8

#     Calculate frequency of ENSO recharge oscillator model
#     fr = calc_fr()
#     pr = (2.*np.pi)/fr

    #Dictionary
    bji_feedbacks = {'bj':bj, 'alphas':alphas_fr, 'ddmc':ddmc_fr, 'zaf':zaf_fr,\
    'ekf':ekf_fr, 'thf':thf_fr}

    bji_coeff = {'mew':mew, 'betau':betau, 'betaw':betaw, 'betauh':betauh,\
    'betah':betah, 'ah':ah, 'thetaae':thetaae, 'tauxat':tauxat,\
    'uae':uae, 'hwwae':hwwae, 'haw':haw, 'hae':hae,\
    'hwtsube':hwtsube, 'h1':h1, 'hm':hm}

    bji_mean = {'u':u, 'v':v, 'w':w, '_dtdxe':dtdxe,\
    '_dtdze':dtdze, 'hwwmhmahe':hwwmhmahe}

    return bji_feedbacks, bji_coeff, bji_mean

#=================================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('basin') #pac or atl
    parser.add_argument('duration') #first,last,all
    parser.add_argument('exp') #Mclm or Oclm
    parser.add_argument('nem') #0-20
    parser.add_argument('n') #Total number of ensembles
    # parser.add_argument('filter') #True or False
    # parser.add_argument('seas') #True or False
    args = parser.parse_args()

    ibas = str(args.basin)
    idur = str(args.duration)
    iexp = str(args.exp)
    inem = int(args.nem)
    itn = int(args.n)
    # ifilter=bool(args.filter)
    # iseas=bool(args.seas)
    print(iexp)
    #=================================================================
    # Extract detrended tropical mean and anomalies from klepto
    # if ifilter==False:
    klepto_atm_detrend_trop = klepto.archives.dir_archive('klepto_atm_detrend_trop',\
    serialized=True, cached=False)
    if iexp=='obs':
        sstm = klepto_atm_detrend_trop[iexp+'_sst_mean'][:-12]
        ssta = klepto_atm_detrend_trop[iexp+'_sst_anom'][:-12]

        tau_xm = klepto_atm_detrend_trop[iexp+'_tau_x_mean'][:-12]
        tau_xa = klepto_atm_detrend_trop[iexp+'_tau_x_anom'][:-12]

        tfluxm = klepto_atm_detrend_trop[iexp+'_tflux_mean'][:-12]
        tfluxa = klepto_atm_detrend_trop[iexp+'_tflux_anom'][:-12]

        um = klepto_atm_detrend_trop[iexp+'_u_mean'][:-12]
        ua = klepto_atm_detrend_trop[iexp+'_u_anom'][:-12]

        vm = klepto_atm_detrend_trop[iexp+'_v_mean'][:-12]
        va = klepto_atm_detrend_trop[iexp+'_v_anom'][:-12]

        wm = klepto_atm_detrend_trop[iexp+'_w_mean'][:-12]
        wa = klepto_atm_detrend_trop[iexp+'_w_anom'][:-12]

        thetam = klepto_atm_detrend_trop[iexp+'_pot_temp_mean'][:-12]
        thetaa = klepto_atm_detrend_trop[iexp+'_pot_temp_anom'][:-12]

        lat=klepto_atm_detrend_trop['obs_lat']
        lon=klepto_atm_detrend_trop['obs_lon']
        lev=klepto_atm_detrend_trop['obs_lev']

    else:
        sstm = klepto_atm_detrend_trop[iexp+'_sst_mean'][inem][:-12]
        ssta = klepto_atm_detrend_trop[iexp+'_sst_anom'][inem][:-12]

        tau_xm = klepto_atm_detrend_trop[iexp+'_tau_x_mean'][inem][:-12]
        tau_xa = klepto_atm_detrend_trop[iexp+'_tau_x_anom'][inem][:-12]

        # evap_heatm = klepto_atm_detrend_trop[iexp+'_evap_heat_mean'][inem]
        # evap_heata = klepto_atm_detrend_trop[iexp+'_evap_heat_anom'][inem]
        # lw_heatm = klepto_atm_detrend_trop[iexp+'_lw_heat_mean'][inem]
        # lw_heata = klepto_atm_detrend_trop[iexp+'_lw_heat_anom'][inem]
        # sens_heatm = klepto_atm_detrend_trop[iexp+'_sens_heat_mean'][inem]
        # sens_heata = klepto_atm_detrend_trop[iexp+'_sens_heat_anom'][inem]
        # swflxm = klepto_atm_detrend_trop[iexp+'_swflx_mean'][inem]
        # swflxa = klepto_atm_detrend_trop[iexp+'_swflx_anom'][inem]
        tfluxm = klepto_atm_detrend_trop[iexp+'_tflux_mean'][inem][:-12]
        tfluxa = klepto_atm_detrend_trop[iexp+'_tflux_anom'][inem][:-12]

        # tfluxm = evap_heatm+lw_heatm+sens_heatm+swflxm
        # tfluxa = evap_heata+lw_heata+sens_heata+swflxa

        # #Extract detrended mld and rho from klepto
        if iexp=='cmip5':
            pass
        else:
            mldm = klepto_atm_detrend_trop[iexp+'_mld_mean'][inem][:-12]
        rhom = klepto_atm_detrend_trop[iexp+'_rho_mean'][inem][:-12]

        um = klepto_atm_detrend_trop[iexp+'_u_mean'][inem][:-12]
        ua = klepto_atm_detrend_trop[iexp+'_u_anom'][inem][:-12]

        vm = klepto_atm_detrend_trop[iexp+'_v_mean'][inem][:-12]
        va = klepto_atm_detrend_trop[iexp+'_v_anom'][inem][:-12]

        if iexp=='cmip5':
            wm = klepto_atm_detrend_trop[iexp+'_w_mean'][inem][:-12]
            wa = klepto_atm_detrend_trop[iexp+'_w_anom'][inem][:-12]
        else:
            wm_ctl1 = klepto_atm_detrend_trop[iexp+'_ctl1_w_mean']
            wm_ctl2 = klepto_atm_detrend_trop[iexp+'_ctl2_w_mean']
            wm_io1 = klepto_atm_detrend_trop[iexp+'_io1_w_mean']
            wm_io2 = klepto_atm_detrend_trop[iexp+'_io2_w_mean']
            wm = np.concatenate((wm_ctl1,wm_ctl2,wm_io1,wm_io2),axis=0)[inem][:-12]

            del wm_ctl1, wm_ctl2, wm_io1, wm_io2
            gc.collect()

            wa_ctl1 = klepto_atm_detrend_trop[iexp+'_ctl1_w_anom']
            wa_ctl2 = klepto_atm_detrend_trop[iexp+'_ctl2_w_anom']
            wa_io1 = klepto_atm_detrend_trop[iexp+'_io1_w_anom']
            wa_io2 = klepto_atm_detrend_trop[iexp+'_io2_w_anom']
            wa = np.concatenate((wa_ctl1,wa_ctl2,wa_io1,wa_io2),axis=0)[inem][:-12]

            del wa_ctl1, wa_ctl2, wa_io1, wa_io2
            gc.collect()

        thetam = klepto_atm_detrend_trop[iexp+'_pot_temp_mean'][inem][:-12]
        thetaa = klepto_atm_detrend_trop[iexp+'_pot_temp_anom'][inem][:-12]

        lat=klepto_atm_detrend_trop['lat']
        lon=klepto_atm_detrend_trop['lon']
        lev=klepto_atm_detrend_trop['lev']

    # # elif ifilter==True:
    # # Extract detrended tropical mean and anomalies from klepto
    # klepto_atm_detrend_trop_flt = klepto.archives.dir_archive('klepto_atm_detrend_trop_flt',\
    # serialized=True, cached=False)
    # if iexp=='obs':
    #     sstm = klepto_atm_detrend_trop_flt[iexp+'_sst_mean'][:-12]
    #     ssta = klepto_atm_detrend_trop_flt[iexp+'_sst_anom'][:-12]
    #
    #     tau_xm = klepto_atm_detrend_trop_flt[iexp+'_tau_x_mean'][:-12]
    #     tau_xa = klepto_atm_detrend_trop_flt[iexp+'_tau_x_anom'][:-12]
    #
    #     tfluxm = klepto_atm_detrend_trop_flt[iexp+'_tflux_mean'][:-12]
    #     tfluxa = klepto_atm_detrend_trop_flt[iexp+'_tflux_anom'][:-12]
    #
    #     um = klepto_atm_detrend_trop_flt[iexp+'_u_mean'][:-12]
    #     ua = klepto_atm_detrend_trop_flt[iexp+'_u_anom'][:-12]
    #
    #     vm = klepto_atm_detrend_trop_flt[iexp+'_v_mean'][:-12]
    #     va = klepto_atm_detrend_trop_flt[iexp+'_v_anom'][:-12]
    #
    #     wm = klepto_atm_detrend_trop_flt[iexp+'_w_mean'][:-12]
    #     wa = klepto_atm_detrend_trop_flt[iexp+'_w_anom'][:-12]
    #
    #     thetam = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_mean'][:-12]
    #     thetaa = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_anom'][:-12]
    #
    #     lat=klepto_atm_detrend_trop_flt['obs_lat']
    #     lon=klepto_atm_detrend_trop_flt['obs_lon']
    #     lev=klepto_atm_detrend_trop_flt['obs_lev']
    #
    # elif iexp=='cmip5':
    #     sstm = klepto_atm_detrend_trop_flt[iexp+'_sst_mean'][inem][:-12]
    #     ssta = klepto_atm_detrend_trop_flt[iexp+'_sst_anom'][inem][:-12]
    #
    #     tau_xm = klepto_atm_detrend_trop_flt[iexp+'_tau_x_mean'][inem][:-12]
    #     tau_xa = klepto_atm_detrend_trop_flt[iexp+'_tau_x_anom'][inem][:-12]
    #
    #     tfluxm = klepto_atm_detrend_trop_flt[iexp+'_tflux_mean'][inem][:-12]
    #     tfluxa = klepto_atm_detrend_trop_flt[iexp+'_tflux_anom'][inem][:-12]
    #
    #     # #Extract detrended mld and rho from klepto
    #     rhom = klepto_atm_detrend_trop_flt[iexp+'_rho_mean'][inem][:-12]
    #
    #     um = klepto_atm_detrend_trop_flt[iexp+'_u_mean'][inem][:-12]
    #     ua = klepto_atm_detrend_trop_flt[iexp+'_u_anom'][inem][:-12]
    #
    #     vm = klepto_atm_detrend_trop_flt[iexp+'_v_mean'][inem][:-12]
    #     va = klepto_atm_detrend_trop_flt[iexp+'_v_anom'][inem][:-12]
    #
    #     wm = klepto_atm_detrend_trop_flt[iexp+'_w_mean'][inem][:-12]
    #     wa = klepto_atm_detrend_trop_flt[iexp+'_w_anom'][inem][:-12]
    #
    #     thetam = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_mean'][inem][:-12]
    #     thetaa = klepto_atm_detrend_trop_flt[iexp+'_pot_temp_anom'][inem][:-12]
    #
    #     lat=klepto_atm_detrend_trop_flt['lat']
    #     lon=klepto_atm_detrend_trop_flt['lon']
    #     lev=klepto_atm_detrend_trop_flt['lev']
    #
    # elif iexp=='Mclm':
    #     sstm = klepto_atm_detrend_trop_flt[iexp+'_sst_mean'][inem][:-12]
    #     ssta = klepto_atm_detrend_trop_flt[iexp+'_sst_anom'][inem][:-12]
    #
    #     tau_xm = klepto_atm_detrend_trop_flt[iexp+'_tau_x_mean'][inem][:-12]
    #     tau_xa = klepto_atm_detrend_trop_flt[iexp+'_tau_x_anom'][inem][:-12]
    #
    #     tfluxm = klepto_atm_detrend_trop_flt[iexp+'_tflux_mean'][inem][:-12]
    #     tfluxa = klepto_atm_detrend_trop_flt[iexp+'_tflux_anom'][inem][:-12]
    #
    #     # #Extract detrended mld and rho from klepto
    #     mldm = klepto_atm_detrend_trop_flt[iexp+'_mld_mean'][inem][:-12]
    #
    #     rhom_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_rho_mean']
    #     rhom_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_rho_mean']
    #     rhom_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_rho_mean']
    #     rhom_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_rho_mean']
    #     rhom = np.concatenate((rhom_ctl1,rhom_ctl2,rhom_io1,rhom_io2),axis=0)[inem][:-12]
    #
    #     del rhom_ctl1, rhom_ctl2, rhom_io1, rhom_io2
    #     gc.collect()
    #
    #     um_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_u_mean']
    #     um_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_u_mean']
    #     um_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_u_mean']
    #     um_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_u_mean']
    #     um = np.concatenate((um_ctl1,um_ctl2,um_io1,um_io2),axis=0)[inem][:-12]
    #
    #     del um_ctl1, um_ctl2, um_io1, um_io2
    #     gc.collect()
    #
    #     ua_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_u_anom']
    #     ua_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_u_anom']
    #     ua_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_u_anom']
    #     ua_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_u_anom']
    #     ua = np.concatenate((ua_ctl1,ua_ctl2,ua_io1,ua_io2),axis=0)[inem][:-12]
    #
    #     del ua_ctl1, ua_ctl2, ua_io1, ua_io2
    #     gc.collect()
    #
    #     vm_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_v_mean']
    #     vm_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_v_mean']
    #     vm_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_v_mean']
    #     vm_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_v_mean']
    #     vm = np.concatenate((vm_ctl1,vm_ctl2,vm_io1,vm_io2),axis=0)[inem][:-12]
    #
    #     del vm_ctl1, vm_ctl2, vm_io1, vm_io2
    #     gc.collect()
    #
    #     va_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_v_anom']
    #     va_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_v_anom']
    #     va_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_v_anom']
    #     va_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_v_anom']
    #     va = np.concatenate((va_ctl1,va_ctl2,va_io1,va_io2),axis=0)[inem][:-12]
    #
    #     del va_ctl1, va_ctl2, va_io1, va_io2
    #     gc.collect()
    #
    #     wm_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_w_mean']
    #     wm_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_w_mean']
    #     wm_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_w_mean']
    #     wm_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_w_mean']
    #     wm = np.concatenate((wm_ctl1,wm_ctl2,wm_io1,wm_io2),axis=0)[inem][:-12]
    #
    #     del wm_ctl1, wm_ctl2, wm_io1, wm_io2
    #     gc.collect()
    #
    #     wa_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_w_anom']
    #     wa_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_w_anom']
    #     wa_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_w_anom']
    #     wa_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_w_anom']
    #     wa = np.concatenate((wa_ctl1,wa_ctl2,wa_io1,wa_io2),axis=0)[inem][:-12]
    #
    #     del wa_ctl1, wa_ctl2, wa_io1, wa_io2
    #     gc.collect()
    #
    #     thetam_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_pot_temp_mean']
    #     thetam_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_pot_temp_mean']
    #     thetam_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_pot_temp_mean']
    #     thetam_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_pot_temp_mean']
    #     thetam = np.concatenate((thetam_ctl1,thetam_ctl2,thetam_io1,thetam_io2),axis=0)[inem][:-12]
    #
    #     del thetam_ctl1, thetam_ctl2, thetam_io1, thetam_io2
    #     gc.collect()
    #
    #     thetaa_ctl1 = klepto_atm_detrend_trop_flt[iexp+'_ctl1_pot_temp_anom']
    #     thetaa_ctl2 = klepto_atm_detrend_trop_flt[iexp+'_ctl2_pot_temp_anom']
    #     thetaa_io1 = klepto_atm_detrend_trop_flt[iexp+'_io1_pot_temp_anom']
    #     thetaa_io2 = klepto_atm_detrend_trop_flt[iexp+'_io2_pot_temp_anom']
    #     thetaa = np.concatenate((thetaa_ctl1,thetaa_ctl2,thetaa_io1,thetaa_io2),axis=0)[inem][:-12]
    #
    #     del thetaa_ctl1, thetaa_ctl2, thetaa_io1, thetaa_io2
    #     gc.collect()
    #
    #     lat=klepto_atm_detrend_trop_flt['lat']
    #     lon=klepto_atm_detrend_trop_flt['lon']
    #     lev=klepto_atm_detrend_trop_flt['lev']
    #
    # #=================================================================
    #Select region for BJI
    if ibas=='atl':
        #Eastern box
        lon1_e = -20%360
        lon2_e = 360

        #western box
        lon1_w = -40%360
        lon2_w = -20%360

        #Tropical box
        lon1_trop = -40%360
        lon2_trop = 360

        lat1 = -5
        lat2 = 5

        lx = 2222000.
        ly = 1111000.

    elif ibas=='pac':
        #Eastern box
        lon1_e = -150%360
        lon2_e = -90%360

        #western box
        lon1_w = 160%360
        lon2_w = -150%360

        #Tropical box
        lon1_trop = 160%360
        lon2_trop = -90%360

        lat1 = -5
        lat2 = 5

        lx = 6667000.
        ly = 1111000.

    #depths
    h0 = 0.   #Surface

    #Calculate depth of mixed layer
    if (iexp=='obs'):
        if ibas=='atl':
            h1 = 32.
        elif ibas=='pac':
            h1 = 50.
        rho=1026.

    elif iexp=='cmip5':
        if ibas=='atl':
            h1 = 38.
        elif ibas=='pac':
            h1 = 45.

        rho=1026.
        # #Calculate density of mixed layer
        # rhom_ebox = np.array(selreg(rhom,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
        # em=False)[3])
        # rho = bn.nanmean(rhom_ebox) #kg/m3

    else:
        #Calculate depth of mixed layer
        mldm_ebox = np.array(selreg(mldm,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,lev1=-999.,lev2=-999.,\
        em=False)[3])
        h1 = bn.nanmean(mldm_ebox)
        #print('H1 is '+str(h1))

        #Calculate density of mixed layer
        rhom_ebox = np.array(selreg(rhom,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
        em=False)[3])
        rho = bn.nanmean(rhom_ebox) #kg/m3

    #Temperature anomalies in the eastern box T
    thetaa_ebox = np.array(selreg(thetaa,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
    em=False)[3])
    #Net surface heat fluxes in the eastern box Q
    tfluxa_ebox = np.array(selreg(tfluxa,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,lev1=-999.,lev2=-999.,\
    em=False)[3])
    #Mean zonal current in the eastern box u|
    lon_ebox,lat_ebox,lev_ebox,um_ebox = selreg(um,lon,lat,lev,\
    lon1_e,lon2_e,lat1,lat2,h0,h1,em=False)
    um_ebox = np.array(um_ebox)
    #Mean meridional current in the eastern box v|
    vm_ebox =np.array(selreg(vm,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
    em=False)[3])
    #Mean upwelling in the eastern box (at the base of the mixed layer) w|
    wm_ebbox = np.array(selreg(wm,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h1,h1,\
    em=False)[3])
    #Mean upwelling in the eastern box (upper ocean) for Hm
    wm_ebox = np.array(selreg(wm,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
    em=False)[3])
    #Anomalous zonal wind stress across the tropics
    tau_xa_tbox = np.array(selreg(tau_xa,lon,lat,lev,lon1_trop,lon2_trop,lat1,lat2,lev1=-999.,lev2=-999.,\
    em=False)[3])
    #Anomalous zonal current in the eastern box
    ua_ebox = np.array(selreg(ua,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
    em=False)[3])
    #Anomalous theta in the western box hw
    thetaa_wubox = np.array(selreg(thetaa,lon,lat,lev,lon1_w,lon2_w,lat1,lat2,lev1=h0,lev2=300.,\
    em=False)[3])
    #Anomalous theta in the eastern box he
    thetaa_eubox = np.array(selreg(thetaa,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,lev1=h0,lev2=300.,\
    em=False)[3])
    #Anomalous upwelling in the eastern box (at the base of the mixed layer)
    wa_ebbox = np.array(selreg(wa,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h1,h1,\
    em=False)[3])
    #Mean temperature in the eastern box T|
    thetam_ebox = np.array(selreg(thetam,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h0,h1,\
    em=False)[3])
    #Anomalous temperature in the eastern box (at the base of the mixed layer) Tsub
    thetaa_ebbox = np.array(selreg(thetaa,lon,lat,lev,lon1_e,lon2_e,lat1,lat2,h1,h1,\
    em=False)[3])

    dx = diff_axis(lon_ebox,axis=0)*111320. #m
    # dy = abs(np.diff(lat_ebox))*110574. #m
    dz = diff_axis(lev_ebox,axis=0) #m

    # n_rm = 30*12 #running mean window

    lat_ebox = np.array(lat_ebox)
    lon_ebox = np.array(lon_ebox)
    lev_ebox = np.array(lev_ebox)

    #=================================================================
    #Calculate 3-month moving average to remove high frequency fluctuations
    rm = 3
    thetaa_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,thetaa_ebox)
    tfluxa_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,tfluxa_ebox)
    tau_xa_tbox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,tau_xa_tbox)
    um_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(um_ebox))
    ua_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(ua_ebox))
    vm_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(vm_ebox))
    wm_ebbox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(wm_ebbox))
    wm_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(wm_ebox))
    wa_ebbox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(wa_ebbox))
    thetaa_wubox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(thetaa_wubox))
    thetaa_eubox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(thetaa_eubox))
    thetam_ebox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(thetam_ebox))
    thetaa_ebbox_rm = np.apply_along_axis(lambda x: movingaverage(x, rm, mode='same'),0,np.array(thetaa_ebbox))

    #=================================================================
    #Mask nan values
    def mask_nan(var):
        var_ma = ma.masked_where(np.nan_to_num(var)==0.,np.nan_to_num(var))
        return var_ma

    thetaa_ebox_ma = mask_nan(thetaa_ebox_rm)
    tfluxa_ebox_ma = mask_nan(tfluxa_ebox_rm)
    tau_xa_tbox_ma = mask_nan(tau_xa_tbox_rm)
    um_ebox_ma = mask_nan(np.array(um_ebox_rm))
    ua_ebox_ma = mask_nan(np.array(ua_ebox_rm))
    vm_ebox_ma = mask_nan(np.array(vm_ebox_rm))
    wm_ebbox_ma = mask_nan(np.array(wm_ebbox_rm))
    wm_ebox_ma = mask_nan(np.array(wm_ebox_rm))
    wa_ebbox_ma = mask_nan(np.array(wa_ebbox_rm))
    thetaa_wubox_ma = mask_nan(np.array(thetaa_wubox_rm))
    thetaa_eubox_ma = mask_nan(np.array(thetaa_eubox_rm))
    thetam_ebox_ma = mask_nan(np.array(thetam_ebox_rm))
    thetaa_ebbox_ma = mask_nan(np.array(thetaa_ebbox_rm))

    #=================================================================
    # #Select season
    # if iseas==True:
    if ibas=='atl': #JJAS
        start=5
    elif ibas=='pac': #NDJF
        start=10
    thetaa_ebox_ma = sel_season(thetaa_ebox_ma, start, n=4, em=False)
    tfluxa_ebox_ma = sel_season(tfluxa_ebox_ma, start, n=4, em=False)
    tau_xa_tbox_ma = sel_season(tau_xa_tbox_ma, start, n=4, em=False)
    um_ebox_ma = sel_season(um_ebox_ma, start, n=4, em=False)
    ua_ebox_ma = sel_season(ua_ebox_ma, start, n=4, em=False)
    vm_ebox_ma = sel_season(vm_ebox_ma, start, n=4, em=False)
    wm_ebbox_ma = sel_season(wm_ebbox_ma, start, n=4, em=False)
    wm_ebox_ma = sel_season(wm_ebox_ma, start, n=4, em=False)
    wa_ebbox_ma = sel_season(wa_ebbox_ma, start, n=4, em=False)
    thetaa_wubox_ma = sel_season(thetaa_wubox_ma, start, n=4, em=False)
    thetaa_eubox_ma = sel_season(thetaa_eubox_ma, start, n=4, em=False)
    thetam_ebox_ma = sel_season(thetam_ebox_ma, start, n=4, em=False)
    thetaa_ebbox_ma = sel_season(thetaa_ebbox_ma, start, n=4, em=False)
    # elif iseas==False:
        # pass

    #=================================================================

    if idur=='all':
        #Calcuate BJI
        bji_feedbacks, bji_coeff, bji_mean = bjerknes_index(thetaa_ebox_ma,tfluxa_ebox_ma,tau_xa_tbox_ma,\
        um_ebox_ma,ua_ebox_ma,vm_ebox_ma,wm_ebbox_ma,wm_ebox_ma,wa_ebbox_ma,\
        thetaa_wubox_ma,thetaa_eubox_ma,thetam_ebox_ma,thetaa_ebbox_ma,
        lat_ebox,lev_ebox,\
        h1,\
        dx,dz,\
        lx,ly,\
        rho)
    elif idur=='first':
        m = int(thetaa_ebox.shape[0]/2)
        bji_feedbacks, bji_coeff, bji_mean = bjerknes_index(thetaa_ebox_ma[:m],tfluxa_ebox_ma[:m],tau_xa_tbox_ma[:m],\
        um_ebox_ma[:m],ua_ebox_ma[:m],vm_ebox_ma[:m],wm_ebbox_ma[:m],wm_ebox_ma[:m],wa_ebbox_ma[:m],\
        thetaa_wubox_ma[:m],thetaa_eubox_ma[:m],thetam_ebox_ma[:m],thetaa_ebbox_ma[:m],
        lat_ebox,lev_ebox,\
        h1,\
        dx,dz,\
        lx,ly,\
        rho)
    elif idur=='last':
        m = int(thetaa_ebox.shape[0]/2)
        bji_feedbacks, bji_coeff, bji_mean = bjerknes_index(thetaa_ebox_ma[m:],tfluxa_ebox_ma[m:],tau_xa_tbox_ma[m:],\
        um_ebox_ma[m:],ua_ebox_ma[m:],vm_ebox_ma[m:],wm_ebbox_ma[m:],wm_ebox_ma[m:],wa_ebbox_ma[m:],\
        thetaa_wubox_ma[m:],thetaa_eubox_ma[m:],thetam_ebox_ma[m:],thetaa_ebbox_ma[m:],
        lat_ebox,lev_ebox,\
        h1,\
        dx,dz,\
        lx,ly,\
        rho)

    #=================================================================
    #Save to klepto
    # if (ifilter==True) and (iseas==False):
    # klepto_bji = klepto.archives.dir_archive('klepto_bji_3rmf',serialized=True, cached=False)
    # elif (ifilter==False) and (iseas==False):
    # klepto_bji = klepto.archives.dir_archive('klepto_bji_3rm',serialized=True, cached=False)
    # elif (ifilter==True) and (iseas==True):
    # klepto_bji = klepto.archives.dir_archive('klepto_bji_3rmf_seas',serialized=True, cached=False)
    # elif (ifilter==False) and (iseas==True):
    klepto_bji = klepto.archives.dir_archive('klepto_bji_3rm_seas',serialized=True, cached=False)

    if inem<=itn-1:
        klepto_bji[str(ibas)+'_'+str(idur)+'_feedbacks_'+str(iexp)+'_'+str(inem+1).zfill(2)] = bji_feedbacks
        klepto_bji[str(ibas)+'_'+str(idur)+'_coeff_'+str(iexp)+'_'+str(inem+1).zfill(2)] = bji_coeff
        klepto_bji[str(ibas)+'_'+str(idur)+'_mean_'+str(iexp)+'_'+str(inem+1).zfill(2)] = bji_mean
    elif inem>=itn:
        klepto_bji[str(ibas)+'_'+str(idur)+'_feedbacks_'+str(iexp)+'T_'+str(inem-itn+1).zfill(2)] = bji_feedbacks
        klepto_bji[str(ibas)+'_'+str(idur)+'_coeff_'+str(iexp)+'T_'+str(inem-itn+1).zfill(2)] = bji_coeff
        klepto_bji[str(ibas)+'_'+str(idur)+'_mean_'+str(iexp)+'T_'+str(inem-itn+1).zfill(2)] = bji_mean
