#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots

"""

__title__ = ""
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

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
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from matplotlib.patches import Polygon

#mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.rcParams['hatch.color'] = '#989898'
mpl.rcParams['hatch.color'] = '#5F45D8'
mpl.rcParams['hatch.linewidth'] = .5
#hatches=['xxxxx']
hatches=['...']
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.weight'] = 'normal'
# mpl.rcParams['font.size'] = 14.
mpl.rcParams['axes.titlepad'] = 3
mpl.rc('text', usetex=False)
#================================================================

#colormaps
cmap_rdybu_r_6 = mpl.colors.ListedColormap(['#0c0003','#4F0012','#d73027','#f46d43','#fdae61','#fee090','#ffffbf',\
'#e0f3f8','#abd9e9','#74add1','#4575b4','#30517D','#313694','#1E215B'][::-1])

cmap_rdbu_r12 = mpl.colors.ListedColormap(['#851108','#992110', '#ac3118', '#c04120', '#d1522b',\
'#de673e', '#ea7c51', '#f59063', '#fea478', '#fcbd9e', '#f9d4c2','#fffaf7',\
'#f8f9fc', '#c7ddfb','#a7cdfe', '#92bcf3', '#80ace4', '#6f9bd6', '#5d8bc8','#4c7bba',\
'#3c6ca9', '#2d5c98', '#1d4e87','#0f3f77'][::-1])

cmap_dry12 = mpl.colors.ListedColormap(['#632e07', '#743d0e','#854b16', '#975a1d', '#a96926','#ba7935',\
'#cc8944', '#dd9a54','#efaa64', '#febd7a', '#fad4ad', '#f7eade','#e1efee', '#b6e4de', '#8ad8ce','#76c7bd',\
'#65b7ad', '#54a69d','#43968c', '#32867d', '#27756d','#1d655d', '#13564e', '#094640'])

cmap_rd_r24 = mpl.colors.ListedColormap(['#4a0000', '#590001', '#680102', '#780204', '#880404', '#980605',\
'#a90804', '#ba0b03', '#cb0e02', '#dd1101', '#ed1a00', '#f13e01', '#f5550a', '#f96918', '#fc7a27', '#ff8b38',\
'#ff9c4c', '#ffac61', '#ffbb78', '#ffca90', '#ffd8aa', '#ffe5c5', '#fff2e1', '#ffffff'][::-1])

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

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def abline(time,slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.arange(len(time))
    y_vals = intercept + slope * x_vals
    return y_vals

def draw_screen_poly( lats, lons, m, ax):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( list(xy), facecolor='none', linestyle='--', edgecolor ='#016c59')
    ax.add_patch(poly)

def plot_map_axesgrid(fig,
var,lon,lat,\
nrows,ncols,\
vmin,vmax,step,mp=0.,
proj=ccrs.Robinson,
u=None,v=None,w_skip=[4,8],\
p=None,\
clim=None,clim_levels=None,fmt='%1.1f',
central_longitude=180.,extent=False,lat1=-90.,lat2=90.,lon1=0.,lon2=360.,lat_step=40,lon_step=120,\
land=True,
title=['Give Subplot Titles Here'],fontsize=10,pad=2,\
main_title='Give Title Here',px=0.5,py=0.82,pxw=0.8,pyw=0.3,al=1,\
ctext='Colorbar text',\
cmap=cmocean.cm.balance):

    projection = proj(central_longitude)
    transform=ccrs.PlateCarree()
    axes_class = (GeoAxes,
    dict(map_projection=projection))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=0.4,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.15,
                    cbar_size='6%',
                    label_mode='')

    for i, ax in enumerate(axgr):
        ax.coastlines(lw=0.8)
        if extent==True:
            ax.set_extent([lon1,lon2,lat1,lat2],crs=transform)
        elif extent==False:
            pass
        if land==True:
            ax.add_feature(cfeature.LAND, facecolor='#B1B1B1')
        elif land==False:
            pass
        yticks = np.arange(lat1, lat2+1, lat_step)
        xticks = np.arange(lon1, lon2+1, lon_step)

        if proj == ccrs.PlateCarree:
            # if nrows < ncols:
            #     if i==0:
            #          ax.set_yticks(yticks, crs=transform)
            #     else:
            #         pass
            #     ax.set_xticks(xticks, crs=transform)
            # elif nrows > ncols:
            #     if (i==nrows*ncols-1):
            #         ax.set_xticks(xticks, crs=transform)
            #     elif (i==0) or (i==1) or (i==2):
            #         ax.set_yticks(yticks, crs=transform)
            #     elif i==3:
            #         ax.set_xticks(xticks, crs=transform)
            #         ax.set_yticks(yticks, crs=transform)
            #     else:
            #         pass
            # elif (nrows == 1) and (ncols == 1):
            ax.set_xticks(xticks, crs=transform)
            ax.set_yticks(yticks, crs=transform)
            ax.tick_params(axis='both', labelsize=8)
            # elif nrows == ncols:
            #     if i==0:
            #         ax.set_yticks(yticks, crs=transform)
            #     elif i==2:
            #         ax.set_xticks(xticks, crs=transform)
            #         ax.set_yticks(yticks, crs=transform)
            #     elif i==3:
            #         ax.set_xticks(xticks, crs=transform)

            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
        else:
            pass

        if title is None:
            pass
        else:
            ax.set_title(title[i],pad=pad)

        levels = np.arange(vmin,vmax,step)

        b = ax.contourf(lon[i], lat[i], var[i],levels=levels,
                    transform=transform,cmap=cmap, extend='both',
                    norm=MidpointNormalize(midpoint=mp,vmin=vmin,vmax=vmax)
                    )

        #Significance
        if p!=None:
            h = ax.contourf(lon[i],lat[i],p[i], colors ='none',es = hatches,alpha=0.6,\
            transform=transform)
        elif p is None:
            pass

        #Plot wind
        if len(u) > 0:
            lons,lats = np.meshgrid(lon[i],lat[i])
            skip=(slice(None,None,w_skip),slice(None,None,w_skip))

            w = ax.quiver(lons[skip], lats[skip], u[i][skip],\
                        v[i][skip],pivot='middle', lw=0.3, color='#1F6E2E',\
                        transform=transform)
        else:
            pass

        if al !=None:
            qk = ax.quiverkey(w, pxw, pyw, al, str(al)+' m s$^{-1}$', labelpos='E',
                       coordinates='figure')
        else:
            pass

        #climatological contours
        if clim!=None:
            X,Y = np.meshgrid(lon[i],lat[i])
            cs = ax.contour(X,Y,clim[i],levels=clim_levels,colors='#016c59',\
            linewidths=0.6,alpha=0.65,transform=transform)
            ax.clabel(cs,inline=1,inline_spacing=1,fmt=fmt,fontsize=4) #'%.0e' %1.1f
        else:
            pass

    #Title, colorbar
    fig.suptitle(main_title,position=(px,py))

    cbar = mpl.colorbar.ColorbarBase(axgr.cbar_axes[0], cmap=cmap,
                               boundaries=levels,
                               orientation='horizontal', extend='both')
    for label in cbar.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    cbar.set_label(ctext)
    return axgr

def plot_map_gridspec(ax,
var,lon,lat,\
vmin,vmax,levels,mp=0.,cmap=cmocean.cm.balance,\
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

    transform=ccrs.PlateCarree()
    ax.coastlines(lw=1.)
    if extent==True:
        ax.set_extent([lon1,lon2,lat1,lat2],crs=transform)
    elif extent==False:
        pass

    if ticks==True:
        #yticks = np.arange(lat1, lat2+1, lat_step)
        yticks = np.arange(lat1,lat2+1,lat_step)
        xticks = np.arange(lon1, lon2+1, lon_step)
        ax.set_xticks(xticks, crs=transform)
        ax.set_yticks(yticks, crs=transform)

        # ax.xaxis.set_tick_params(labelsize=fontsize)
        # ax.yaxis.set_tick_params(labelsize=fontsize)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    else:
        pass

    b = ax.contourf(lon, lat, var,levels=levels,
                transform=transform,cmap=cmap, extend='both',
                norm=MidpointNormalize(midpoint=mp,vmin=vmin,vmax=vmax)
                )
    #Significance
    if len(p) > 0:
        #var_sig = ma.masked_where(p > 0.05, p)
        #var_sig = ma.masked_where(var_sig == np.nan, var_sig)
        h = ax.contourf(lon,lat,p, colors = 'none', hatches = hatches,\
        transform=transform)
    else:
        pass

    #Plot wind
    if len(u1) > 0:
        #All wind
        lons,lats = np.meshgrid(lon,lat)
        skip=(slice(None,None,w_skip),slice(None,None,w_skip))

        # w = m.quiver(x[skip], y[skip], u[skip],\
        #             v[skip],pivot='middle', lw=0.2, color='grey')
        #significant wind
        w1 = ax.quiver(lons[skip], lats[skip], u1[skip],\
                    v1[skip],pivot='middle', lw=0.1, color='#525252',alpha=0.8,\
                    transform=transform, scale=scale)
        w2 = ax.quiver(lons[skip], lats[skip], u2[skip],\
                    v2[skip],pivot='middle', lw=0.1, color=cv,\
                    transform=transform,scale=scale)
        qk = ax.quiverkey(w2, pxw, pyw, al, str(al)+' '+str(w_unit), labelpos='E',
                   coordinates='figure')
    else:
        pass

    # if al > 0:
    #     qk = ax.quiverkey(w1, pxw, pyw, al, str(al)+' '+str(w_unit), labelpos='E',
    #                coordinates='figure')
    # else:
    #     pass

    #climatological contours
    if np.ndim(clim) > 1:
        X,Y = np.meshgrid(lon,lat)
        cs = ax.contour(X,Y,clim,levels=clim_levels,colors=cc,
        linewidths=0.6,transform=transform)
        ax.clabel(cs,inline=1,inline_spacing=1,fontsize=fontsize,fmt=fmt) #'%1.1f' '%.0e'
    else:
        pass

    if land==True:
        ax.add_feature(cfeature.LAND, facecolor='#B1B1B1')
    elif land==False:
        pass

    ax.set_title(title,pad=pad,loc=loc_title)


def plot_map_gridspec_basemap(m,ax,
var,lon,lat,mask_io,\
vmin,vmax,levels,mp=0.,cmap=cmocean.cm.balance,\
central_longitude=180.,\
extent=False,lat1=-90.,lat2=90.,lon1=0.,lon2=360.,lat_step=10,lon_step=60,\
land=True,cl='#B1B1B1',
yticks=True,xticks=True,\
u1=None,v1=None,u2=None,v2=None,w_skip=[4,8],scale=1.,cv='k',
p=None,\
clim=None,clim_levels=None,fmt='%1.1f',cc='#2A7F53',
oc=True,
title=['Give Subplot Titles Here'],fontsize=10,pad=2,loc_title='center',\
px=0.5,py=0.82,pxw=0.8,pyw=0.3,al=1,w_unit='m s$^{-1}$ decade$^{-1}$'
):

    m.drawcoastlines(linewidth=1.)

    if land==True:
        m.fillcontinents(color=cl)
    elif land==False:
        pass

    #ticks
    xtick = np.arange(lon1, lon2+1, lon_step)
    ytick = np.arange(lat1,lat2+1,lat_step)
    if yticks == True:
        try:
            m.drawparallels(ytick,labels=[1,0,0,0],fontsize=fontsize,linewidth=0.,xoffset=0.5)
        except ValueError:
            pass
    else:
        m.drawparallels(ytick,labels=[0,1,1,0],fontsize=fontsize,linewidth=0.,xoffset=0.5)

    if xticks == True:
        mer = m.drawmeridians(xtick,labels=[0,0,0,1],fontsize=fontsize,linewidth=0.)
    else:
        mer = m.drawmeridians(xtick,labels=[1,1,0,0],fontsize=fontsize,linewidth=0.)

    lons,lats = np.meshgrid(lon,lat)
    x, y = m(lons,lats)
    b = m.contourf(x, y, var,levels=levels,
                cmap=cmap,extend='both',
                norm=MidpointNormalize(midpoint=mp,vmin=vmin,vmax=vmax)
                )
    if oc==True:
        m.contour(x,y,mask_io,levels=1,colors='#01665e')
    else:
        pass

    #Significance
    if len(p) > 0:
        #var_sig = ma.masked_where(p > 0.05, p)
        #var_sig = ma.masked_where(var_sig == np.nan, var_sig)
        h = m.contourf(x,y,p, colors = 'none', hatches = hatches)
    else:
        pass

    #climatological contours
    if np.ndim(clim) > 1:
        cs = m.contour(x,y,clim,levels=clim_levels,colors=cc,
        linewidths=1.)
        plt.clabel(cs,inline=1,inline_spacing=1,fontsize=fontsize,fmt=fmt) #'%1.1f' '%.0e'
    else:
        pass

    #Plot wind
    if len(u1) > 0:
        #All wind
        skip=(slice(None,None,w_skip),slice(None,None,w_skip))

        x,y=x[:,:-1],y[:,:-1]
        #significant wind
        w1 = m.quiver(x[skip], y[skip], u1[skip],\
                    v1[skip],pivot='middle', lw=0.1, color='#525252',alpha=0.8, scale=scale)
        w2 = m.quiver(x[skip], y[skip], u2[skip],\
                    v2[skip],pivot='middle', lw=0.1, color=cv,scale=scale)
        qk = ax.quiverkey(w2, pxw, pyw, al, str(al)+' '+str(w_unit), labelpos='E',
                   coordinates='figure')

    ax.set_title(title,pad=pad,loc=loc_title)


class OOMFormatter(mpl.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mpl.ticker._mathdefault(self.format)


def plot_hov(ax,levels,x,y,z,p,clim,clim_levels,cmap,xtick1,xtick2,xtick_step,\
ytick1,ytick2,ytick_step,ylim1,ylim2,xlim1,xlim2,yticks_label,xlabel,ylabel,title,\
fontsize,pad,u1,v1,u2,v2,ctext,main_title,custom_label=True,invert_yaxis=False,\
w_skip1=8, w_skip2=8, scale=0.5, atm=True,\
px=0.5,py=0.82,pxw=0.45,pyw=0.16,al=-0.1,w_unit='m s$^{-1}$ decade$^{-1}$'
):
    X,Y = np.meshgrid(x,y)
    #contour plot
    h = ax.contourf(X,Y,z,cmap=cmap,levels=levels,extend='both')

    #Plot arrows
    if len(u1) > 0:
        skip=(slice(None,None,w_skip1),slice(None,None,w_skip2))

        w1 = ax.quiver(X[skip], Y[skip], u1[skip],v1[skip],pivot='middle', lw=0.1,\
                    color='k',alpha=0.5, scale=scale)
        w2 = ax.quiver(X[skip], Y[skip], u2[skip],v2[skip],pivot='middle', lw=0.1,\
                    color='k', scale=scale)
        qk = ax.quiverkey(w2, pxw, pyw, al, str(al)+' '+str(w_unit), labelpos='E',
                   coordinates='figure')
    else:
        pass
    #climatological contours
    if clim is None:
        pass
    else:
        cs = ax.contour(X,Y,clim,levels=clim_levels,colors='grey',linewidths=1.)
        ax.clabel(cs,inline=1,inline_spacing=1,fontsize=fontsize,fmt='%.0f') #'%.0e'

    #Significance
    if p is None:
        pass
    else:
        s = ax.contourf(X,Y,p, colors = 'none', hatches = hatches)

    #subplot title
    ax.set_title(title,pad=pad, loc='left')
    #Axis ticks
    ax.set_xticks(np.arange(xtick1,xtick2,xtick_step))
    ax.set_xlim([xlim1,xlim2])

    # #Change to degrees longitude
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # if xlabel == 'Longitude':
    #     labels = ['120\N{DEGREE SIGN}E','150\N{DEGREE SIGN}E',\
    #     '180\N{DEGREE SIGN}','150\N{DEGREE SIGN}W','120\N{DEGREE SIGN}W',\
    #     '90\N{DEGREE SIGN}W','60\N{DEGREE SIGN}W'] #,'30\N{DEGREE SIGN}W','0\N{DEGREE SIGN}']
    # elif xlabel == 'Latitude':
    #     labels = ['90\N{DEGREE SIGN}S','60\N{DEGREE SIGN}S','30\N{DEGREE SIGN}S','0\N{DEGREE SIGN}','30\N{DEGREE SIGN}N','60\N{DEGREE SIGN}N','90\N{DEGREE SIGN}N']
    # ax.set_xticklabels(labels)

    ax.set_yticks(np.arange(ytick1,ytick2,ytick_step))
    ax.set_ylim([ylim1,ylim2])
    ax.minorticks_off()
    ax.set_ylabel(ylabel)

    if custom_label==True:
        ax.set_yticklabels(yticks_label)
    else:
        pass
    if invert_yaxis==True:
        ax.invert_yaxis()
    else:
        pass

    if atm==True:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(mticker.FixedLocator([200,500,850]))
        ax.yaxis.set_minor_locator(mticker.FixedLocator([]))
    else:
        pass

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    #
    #fig.text(0.5, 0.03, xlabel, ha='center', fontsize=fontsize)
    #fig.text(0.03, 0.5, ylabel, va='center', rotation='vertical', fontsize=fontsize)

    # fig.subplots_adjust(right=0.8, hspace=0.35)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    # cb = fig.colorbar(h, cax=cbar_ax)
    # for label in cbar_ax.yaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)
    # cb.set_label(ctext)
    #fig.suptitle(main_title,fontsize=10)
