#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Test

"""

__title__ = "Monte Carlo Test of SST trends using Control Simulation"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#==============================================================================
#Load modules
import shelve
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

#My modules
from trend import sel_region

#==============================================================================

def pdf(sst,lon,lat):
    #select Region
    sst_reg = sel_reg(lon,lat,sst)
    sst_reg = sst_reg.mean(axis=1).mean(axis=1)
    #Bootstrap blocks
    n = 30 #Number of years
    reps = 10000 #Repetitions
    #Randomly bootstrap blocks
    sst_boot = np.random.choice(sst_reg, (n,reps),replace =True)
    #Calculate trend for each block
    slope_boot = np.zeros(sst_boot[0].shape)
    for i in range(len(sst_boot[0])):
        slope_boot[i],intercept, r_value, p_value, std_err = \
        stats.linregress(range(len(sst_boot[:,i])),sst_boot[:,i])
    return slope_boot

def trend_var(sst_t,lon,lat):
    sst_reg = sel_region(lon,lat,sst_t)
    sst_reg = sst_reg.mean(axis=1).mean(axis=1)
    slope_var, intercept, r_value, p_value, std_err = \
    stats.linregress(range(len(sst_reg)),sst_reg)
    return slope_var

def pdf_plot():
    #Plot the PDF
    slope_boot_con = pdf(sst_con,lon,lat)
    slope_boot_ff = pdf(sst_ff,lon,lat)
    sns.kdeplot(slope_boot_con,color='teal', shade=True)
    sns.kdeplot(slope_boot_ff,color='#4d4d4d', shade=True)
    #Demaracte the tails
    lower_con, upper_con = np.percentile(slope_boot_con, [2.5, 97.5], axis=0)
    plt.axvline(upper_con,c='teal',ls='--')
    plt.axvline(lower_con,c='teal',ls='--')
    lower_ff, upper_ff = np.percentile(slope_boot_ff, [2.5, 97.5], axis=0)
    plt.axvline(upper_ff,c='#4d4d4d',ls='--')
    plt.axvline(lower_ff,c='#4d4d4d',ls='--')
    #Plot individual trends
    color_list = {'FF':'#4d4d4d','GHG':'#ef8a62','OA':'#4575b4'}
    #1950
    plt.axvline(trend_var(sst_ff_1950,lon,lat), c = color_list['FF'], ls='--', lw = 2.5)
    plt.axvline(trend_var(sst_ghg_1950,lon,lat), c = color_list['GHG'], ls='--', lw = 2.5)
    plt.axvline(trend_var(sst_oa_1950,lon,lat), c = color_list['OA'], ls='--', lw = 2.5)
    #1975
    plt.axvline(trend_var(sst_ff_1975,lon,lat), c = color_list['FF'], ls='-', lw = 2.5)
    plt.axvline(trend_var(sst_ghg_1975,lon,lat), c = color_list['GHG'], ls='-', lw = 2.5)
    plt.axvline(trend_var(sst_oa_1975,lon,lat), c = color_list['OA'], ls='-', lw = 2.5)
    #Plot labels and save figure
    plt.xlabel('Trend');
    plt.ylabel('Density function');
    plt.ticklabel_format(style='sci', scilimits = (0,0))
    #save figure
    plt.savefig('kde_all_trend_yly.png', dpi=300)
    plt.show()

#==============================================================================
if __name__ == "__main__":
    #Load Variables
    #SST anomalies Control
    with shelve.open('sst_yly', 'r') as shelf:
        sst_con = shelf['sst']
        lat = shelf['lat']
        lon = shelf['lon']
    #SST anomalies
    #Full forcing
    with shelve.open('./../full_forcing/sst_yly', 'r') as shelf:
        sst_ff = shelf['sst']
        sst_ff_1950 = shelf['sst_1950']
        sst_ff_1975 = shelf['sst_1975']
    #GHG
    with shelve.open('./../GHG/sst_yly', 'r') as shelf:
        sst_ghg_1950 = shelf['sst_1950']
        sst_ghg_1975 = shelf['sst_1975']
    #OA
    with shelve.open('./../Ozone_Aerosols/sst_yly', 'r') as shelf:
        sst_oa_1950 = shelf['sst_1950']
        sst_oa_1975 = shelf['sst_1975']

    pdf_plot()
