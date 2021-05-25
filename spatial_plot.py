#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Plots

"""

__title__ = ""
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#==============================================================================
#General modules
import numpy as np
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

#mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.rcParams['hatch.color'] = '#989898'
mpl.rcParams['hatch.color'] = '#5F45D8'
mpl.rcParams['hatch.linewidth'] = .5
#hatches=['xxxxx']
hatches=['...']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['font.size'] = 14.
mpl.rcParams['axes.titlepad'] = 2
mpl.rc('text', usetex=False)
#================================================================

#colormaps
cmap_rdbu_r12 = mpl.colors.ListedColormap(['#851108','#992110', '#ac3118', '#c04120', '#d1522b',\
'#de673e', '#ea7c51', '#f59063', '#fea478', '#fcbd9e', '#f9d4c2','#fffaf7',\
'#f8f9fc', '#c7ddfb','#a7cdfe', '#92bcf3', '#80ace4', '#6f9bd6', '#5d8bc8','#4c7bba',\
'#3c6ca9', '#2d5c98', '#1d4e87','#0f3f77'][::-1])

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def plot_map_gridspec(ax,
var,lon,lat,\
vmin,vmax,levels,mp=0.,cmap=cmap_rdbu_r12,\
central_longitude=180.,\
extent=False,lat1=-90.,lat2=90.,lon1=0.,lon2=360.,lat_step=10,lon_step=60,\
ticks=True,\
u1=None,v1=None,u2=None,v2=None,w_skip=[4,8],scale=1,cv='k',
p=None,\
clim=None,clim_levels=None,fmt='%1.1f',cc='#2A7F53',
land=True,
title=['Give Subplot Titles Here'],fontsize=12,pad=2,loc_title='left',\
px=0.5,py=0.82,pxw=0.8,pyw=0.3,al=1,w_unit='m s$^{-1}$ decade$^{-1}$'
):
"""
ax: gridspec axis
var: 2-D array
lon,lat: 1-D array
vmin,vmax,levels: Minimum and maximum value, levels
mp: Midpoint on the colorbar
cmap: colormap
extent,lat1,lat2,lon1,lon2: Selects regional extent, bounds
lat_step,lon_step: tick frequency
u1,v1: vectors
u2,v2: significant vectors
p: 2-D array of significance value
clim, clim_levels: 2-D array of climatological contours, levels
cc: contour color
"""
    #Axis transform
    transform=ccrs.PlateCarree()

    #linewidth of coastlines
    ax.coastlines(lw=1.)

    #Regional extent
    if extent==True:
        ax.set_extent([lon1,lon2,lat1,lat2],crs=transform)
    elif extent==False:
        pass

    #Axis ticks
    if ticks==True:
        yticks = np.arange(lat1,lat2+1,lat_step)
        xticks = np.arange(lon1, lon2+1, lon_step)
        ax.set_xticks(xticks, crs=transform)
        ax.set_yticks(yticks, crs=transform)

        #Format tick labels
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    else:
        pass

    #Plot contourf
    b = ax.contourf(lon, lat, var,levels=levels,
                transform=transform,cmap=cmap, extend='both',
                norm=MidpointNormalize(midpoint=mp,vmin=vmin,vmax=vmax)
                )

    #Significance (stippling)
    if len(p) > 0:
        h = ax.contourf(lon,lat,p, colors = 'none', hatches = hatches,\
        transform=transform)
    else:
        pass

    #Plot wind vectors
    if len(u1) > 0:
        #All wind
        lons,lats = np.meshgrid(lon,lat)
        skip=(slice(None,None,w_skip),slice(None,None,w_skip))

        w = m.quiver(x[skip], y[skip], u[skip],\
                     v[skip],pivot='middle', lw=0.2, color='grey')

        #Significant wind
        w1 = ax.quiver(lons[skip], lats[skip], u1[skip],\
                    v1[skip],pivot='middle', lw=0.1, color='#525252',alpha=0.8,\
                    transform=transform, scale=scale)
        w2 = ax.quiver(lons[skip], lats[skip], u2[skip],\
                    v2[skip],pivot='middle', lw=0.1, color=cv,\
                    transform=transform,scale=scale)
        #Quiver key
        qk = ax.quiverkey(w2, pxw, pyw, al, str(al)+' '+str(w_unit), labelpos='E',
                   coordinates='figure')
    else:
        pass

    #climatological contours
    if np.ndim(clim) > 1:
        X,Y = np.meshgrid(lon,lat)
        cs = ax.contour(X,Y,clim,levels=clim_levels,colors=cc,
        linewidths=0.6,transform=transform)
        ax.clabel(cs,inline=1,inline_spacing=1,fontsize=fontsize,fmt=fmt) #'%1.1f' '%.0e'
    else:
        pass

    #Fill land
    if land==True:
        ax.add_feature(cfeature.LAND, facecolor='#B1B1B1')
    elif land==False:
        pass

    #set title
    ax.set_title(title,pad=pad,loc=loc_title)
