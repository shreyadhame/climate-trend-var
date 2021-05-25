#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Model evaluation for seasonality, growth/decay"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#==============================================================================
#Load modules
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde, stats
from spectrum import dpss, pmtm

def make_xarray_groupby_seasons(ind,time):
    ind_xr = xr.DataArray(data=ind,coords=time.coords,dims=time.dims)
    ind_s_std = ind_xr.groupby('time.month').std()
    return ind_s_std

def groupby_months(ind):
    split = np.split(ind,len(ind)//12)
    stack = np.dstack(split).squeeze()
    seas = np.nanstd(stack,axis=-1)
    return seas

def plot_kde_scatter(x,y,cmap,ax,alpha):
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=20, cmap=cmap, alpha=alpha)

def plot_polyfit(x,c,ax,label,px,py):
    X_test = np.linspace(-3.5, 3.5, 100)

    if x.ndim==3:
        u = [np.polyfit(x[i,:,0], x[i,:,1], deg=2) for i in range(len(x))]
        for i in range(len(u)):
            if u[i][0] < 0.:
                pass
            else:
                u[i][0] = u[i][0]*(-1.)
        y_pred = np.stack([u[i][0] * X_test**2 + u[i][1]*X_test + u[i][2] for i in range(len(x))])

        ax.plot(X_test,y_pred.mean(axis=0),c=c,label=label,lw=2.5)
        ax.fill_between(X_test,y_pred.mean(axis=0)-y_pred.std(axis=0),y_pred.mean(axis=0)+y_pred.std(axis=0),\
                        label=label,color=c,alpha=0.5)
        ax.text(px,py,'α = '+str(np.round(u[0].mean(),3))+' '+u"\u00B1"+' '+str(np.round(u[0].std(),2)),c=c)
    elif x.ndim==2:
        u = np.polyfit(x[:,0], x[:,1], deg=2)
        y_pred = u[0] * X_test**2 + u[1]*X_test + u[2]
        ax.plot(X_test,y_pred,c=c,label=label,lw=2.5)
        ax.text(px,py,'α = '+str(np.round(u[0],3)),c=c)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right',frameon=False)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

def psd(data):
    N = len(data)
    NW=4 #haf bandwidth parameter 2.5, 3, 3.5, 4
    # k=4
    dt=1
    [tapers, eigen] = dpss(N, NW)
    Sk_complex, weights, eigenvalues=pmtm(data, e=eigen, v=tapers, NFFT=N, show=False)

    Sk = abs(Sk_complex)**2
    Sk = np.mean(Sk * np.transpose(weights), axis=0) * dt
    Sk = Sk/np.max(Sk)
    return Sk

def calc_rs_growthdecay(ts):
    """
    Input:-
    ts: 1-D array var
    start: index of December month

    Output: regression coefficient time series

    """
    start=11
    ts_peak = ts[:-12][start::12]

    ts_s1 = []
    for i in range(start+1):
        ts_s1.append(ts[:-12][i::12])
    ts_s1 = np.stack(ts_s1)

    ts_s2 = []
    for i in range(start+1):
        ts_s2.append(ts[12:][i::12])
    ts_s2 = np.stack(ts_s2)

    ts_m = np.concatenate((ts_s1, ts_peak[np.newaxis,:], ts_s2))

    r = np.zeros(ts_m.shape[0])
    for i in range(len(r)):
        r[i] = stats.theilslopes(ts_m[i], ts_peak)[0]

    return r
