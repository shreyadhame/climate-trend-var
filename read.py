#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read netCDF file

"""

__title__ = "Read a netCDF file, shift longitude, select time and region"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#===============================================================

#General modules
import argparse
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.colors as colors

#====================================================

def wgtvar(var, lon, lat, lev, em=False):
    if em==True:
        if var.ndim==5: #(em, time, lev, lat, lon)
            wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180,(len(lev),1,len(lon))))[np.newaxis,np.newaxis,...]
        elif var.ndim==4: #(em, time, lat,lon) or (time,em,lat,lon)
            wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180,(1,len(lon))))[np.newaxis,np.newaxis,...]
        var_mean = np.zeros((var.shape[0],var.shape[1]))
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                var_mean[i,j] = np.sum(var[i,j] * wgtmat * ~var.mask[i,j])/np.sum(wgtmat * ~var.mask[i,j])

    elif em==False:
        if var.ndim ==4: #(time, lev, lat, lon) or #(time,ensemble,lat,lon)
            wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180,(len(lev),1,len(lon))))[np.newaxis,...]
        elif var.ndim == 3: #(time, lat, lon)
            wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180,(1,len(lon))))[np.newaxis,...]
        var_mean = np.zeros(var.shape[0])
        for i in range(var.shape[0]):
            var_mean[i] = np.sum(var[i] * wgtmat * ~var.mask[i])/np.sum(wgtmat * ~var.mask[i])

        if var.ndim == 2: #(lat,lon)
            wgtmat = np.cos(np.tile(abs(lat[:,None])*np.pi/180,(1,len(lon))))
            var_mean = np.sum(var * wgtmat * ~var.mask)/np.sum(wgtmat * ~var.mask)
    return var_mean

def shift_lons(ds, lon_dim='longitude'):
    """ Shift longitudes from [-180, 180] to [0, 360] """
    lons = ds[lon_dim].values
    new_lons = np.empty_like(lons)
    mask = lons < 180
    new_lons[mask] = lons[mask]%360
    ##Shift longitudes from [0, 360] to [-180, 180]
    # mask = lons > 180
    # new_lons[mask] = -(360. - lons[mask])
    new_lons[~mask] = lons[~mask]
    ds[lon_dim].values = new_lons
    return ds

def roll_data(var,lon):
    #Find the longitude resolution or step difference
    lon_step = abs(lon[1]-lon[0])
    #Find the index in the array from where rolling will begin
    roll=[]
    actual_roll = []
    for i in range(len(lon)-1):
        #if i < lon[-1]:
            if abs(lon[i+1] - lon[i]) > lon_step +1.e-17:
                roll = i
                break
            else:
                roll = 0.
    if roll ==0:
        actual_roll = len(lon) - roll
    else:
        actual_roll = len(lon) - roll -1
    lon_roll = lon.roll(longitude=actual_roll)
    var_roll = var.roll(longitude=actual_roll)
    return lon_roll, var_roll

def read_data(ipath, ivar, imask=None, decode_times=True, start_time=850, end_time=2005):
    """ Reads variables of NETCDF file"""
    #Open dataset
    with xr.open_dataset(ipath,decode_times=decode_times) as ds:
    #Change longitude and latitude dimnames for consistency among datasets
        #rename time
        try:
            ds.rename({'t':'time'},inplace=True)
        except ValueError:
            pass
        #rename lonlat
        try:
            ds.rename({'lat':'latitude','lon':'longitude'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'LAT':'latitude','LON':'longitude'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'LATITUDE':'latitude','LONGITUDE_0':'longitude'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'latitude':'latitude','longitude_0':'longitude'},inplace=True)
        except ValueError:
            pass
        if ivar=='u' or ivar=='v' or ivar=='tau_x':
            try:
                ds.rename({'xu_ocean':'longitude','yu_ocean':'latitude'},inplace=True)
            except ValueError:
                pass
        else:
            try:
                ds.rename({'xt_ocean':'longitude','yt_ocean':'latitude'},inplace=True)
            except ValueError:
                pass
        try:
            ds.rename({'XU_OCEAN':'longitude','YU_OCEAN':'latitude'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'XU_OCEAN':'longitude','YU_OCEAN1':'latitude'},inplace=True)
        except ValueError:
            pass
        #rename levels
        try:
            ds.rename({'level':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'z16_p_level':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'st_ocean':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'ST_OCEAN1_30':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'ST_OCEAN':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'plev':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'pressure':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'lev_2':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'Depth':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'DEPTH':'lev'},inplace=True)
        except ValueError:
            pass

        #extract variable
        var = ds[ivar].squeeze()

        #Extract lonlat, lev, time
        lat = ds.latitude
        try:
            lon = ds.longitude
            lon_dim = lon.dims[0]
        except AttributeError:
            lon=[]
        #No levels
        try:
            lev=ds.lev
        except AttributeError:
            lev=[]

        ds = xr.decode_cf(ds,decode_times=decode_times)
        try:
            time=ds.time
        except AttributeError:
            time=[]

        #Fix time for CESM-LME
        # if lme == 'True':
        #     years = np.arange(start_time,end_time,10000)
        #     years = np.repeat(years,12)
        #     nyears = int((end_time/10000.)-(start_time/10000.))
        #     months = np.arange(100,1300,100)
        #     months = np.tile(months,nyears)
        #     days = np.repeat(1,len(time))
        #     dates = years + months + days
        #     datetimes = pd.to_datetime(dates, format='%Y%m%d', errors='ignore')
        #     time = []
        #     for i in range(len(datetimes)):
        #          time.append(str(datetimes[i].year).zfill(4)+'-'+str(datetimes[i].month).zfill(2))
        # elif lme == 'False':
        #     pass
        #Roll data and longitude

        # #select_lev
        # if select_lev == True:
        #     lev = lev.sel(lev=slice(lev1,lev2))
        #     var = var.sel(lev=slice(lev1,lev2))
        # else:
        #     pass
        #flip lat
        if lat[0] > 0:
            lat = np.flip(lat)
            var = np.flip(var,axis=-2)
        else:
            pass
        #shift lon
        if lon[0]<0:
            ds = shift_lons(ds, lon_dim=str(lon_dim))
            lon_roll,var_roll = roll_data(var,ds.longitude)
        else:
            lon_roll = lon
            var_roll = var

    #Basin Mask
    if imask is None:
        basin_mask = []
    else:
        basin_mask = ds[imask]
        if lon[0]<0:
            basin_mask = roll_data(basin_mask,lon)[1]
        else:
            pass
    return lon_roll,lat,lev,var_roll,time,basin_mask

def read_mdata(ipath, ivar, imask=None, decode_times=True, concat_dim='ensemble',start_time=850, end_time=2005):
    """ Reads variables of NETCDF file"""
    #Open dataset
    with xr.open_mfdataset(ipath,decode_times=decode_times,concat_dim=concat_dim) as ds:
    #Change longitude and latitude dimnames for consistency among datasets
        #rename time
        try:
            ds.rename({'t':'time'},inplace=True)
        except ValueError:
            pass
        #rename lonlat
        try:
            ds.rename({'lon':'longitude','lat':'latitude'},inplace=True)
        except ValueError:
            pass
        if ivar=='u' or ivar=='v' or ivar=='tau_x':
            try:
                ds.rename({'xu_ocean':'longitude','yu_ocean':'latitude'},inplace=True)
            except ValueError:
                pass
        else:
            try:
                ds.rename({'xt_ocean':'longitude','yt_ocean':'latitude'},inplace=True)
            except ValueError:
                pass
        try:
            ds.rename({'XU_OCEAN':'longitude','YU_OCEAN':'latitude'},inplace=True)
        except ValueError:
            pass
        #rename levels
        try:
            ds.rename({'level':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'z16_p_level':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'st_ocean':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'ST_OCEAN':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'plev':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'pressure':'lev'},inplace=True)
        except ValueError:
            pass
        try:
            ds.rename({'lev_2':'lev'},inplace=True)
        except ValueError:
            pass

        #Extract lonlat, lev, time
        lat = ds.latitude
        lon = ds.longitude
        lon_dim = lon.dims[0]
        #No levels
        try:
            lev=ds.lev
        except AttributeError:
            lev=[]

        ds = xr.decode_cf(ds)
        try:
            time=ds.time
        except AttributeError:
            time=[]

        #extract variable
        var = ds[ivar].squeeze()

        #flip lat
        if lat[0] > 0:
            lat = np.flip(lat)
            var = np.flip(var,axis=-2)
        #shift lon
        if lon[0]<0:
            ds = shift_lons(ds, lon_dim=str(lon_dim))
            lon_roll,var_roll = roll_data(var,ds.longitude)
        else:
            lon_roll = lon
            var_roll = var
        ##Changing time units (ex. when using CESM-LME)
        # ds.time.attrs['units'] = u'days since 0000-00-01 00:00:00
        # ds = xr.decode_cf(ds)
    #Basin Mask
    if imask is None:
        basin_mask = []
    else:
        basin_mask = ds[imask]
        if lon[0]<0:
            basin_mask = roll_data(basin_mask,lon)[1]
        else:
            pass
    return lon_roll,lat,lev,var_roll,time,basin_mask

def seltime(var,time,start_yr,end_yr):
    """ Select time period of analysis """
    var_t = var.sel(time=slice(str(start_yr),str(end_yr)))
    time_t = time.sel(time=slice(str(start_yr),str(end_yr)))
    return var_t, time_t

def selreg_xr(var, lon, lat, lev, lon1, lon2, lat1, lat2, lev1, lev2):
    """ Select region of analysis """
    lat_1 = lat.sel(latitude=lat1,method='nearest')
    lat_2 = lat.sel(latitude=lat2,method='nearest')
    #CAUTION: Nearest method will not work for longitude if you have rolled it since \
    #coordinates in xarray are immutable!
    lon_1 = lon.sel(longitude=lon1,method='nearest')
    lon_2 = lon.sel(longitude=lon2,method='nearest')

    if var.ndim == 4:
        lev_1 = lev.sel(longitude=lev1,method='nearest')
        lev_2 = lev.sel(longitude=lev2,method='nearest')
        levs = lev.sel(lev=slice(lev_1,lev_2))
        if lat[0] < 0:
            box = var.sel(latitude=slice(lat_1,lat_2),longitude=slice(lon_1,lon_2),\
            lev=slice(lev_1,lev_2))
            lats = lat.sel(latitude=slice(lat_1,lat_2))
        elif lat[0] > 0:
            box = var.sel(latitude=slice(lat_2,lat_1),longitude=slice(lon_1,lon_2),\
            lev=slice(lev_1,lev_2))
            lats = lat.sel(latitude=slice(lat_2,lat_1))
    else:
        if lat[0] < 0:
            box = var.sel(latitude=slice(lat_1,lat_2),longitude=slice(lon_1,lon_2))
            lats = lat.sel(latitude=slice(lat_1,lat_2))
        elif lat[0] > 0:
            box = var.sel(latitude=slice(lat_2,lat_1),longitude=slice(lon_1,lon_2))
            lats = lat.sel(latitude=slice(lat_2,lat_1))
        levs=[]
    lons = lon.sel(longitude=slice(lon_1,lon_2))
    return lons,lats,levs,box

def selreg(var, lon, lat, lev, lon1, lon2, lat1, lat2, lev1, lev2, em):
    ind_start_lat=int(np.abs(lat-(lat1)).argmin())
    ind_end_lat=int(np.abs(lat-(lat2)).argmin())+1
    ind_start_lon=int(np.abs(lon-(lon1)).argmin())
    ind_end_lon=int(np.abs(lon-(lon2)).argmin())+1

    #lonlat
    lons = lon[ind_start_lon:ind_end_lon]
    lats = lat[ind_start_lat:ind_end_lat]
    if em==True:
        if lev1>=0.:
            if var.ndim==5: #(em,time,lev,lat,lon)
                ind_start_lev=int(np.abs(lev-(lev1)).argmin())
                ind_end_lev=int(np.abs(lev-(lev2)).argmin())+1
                levs = lev[ind_start_lev:ind_end_lev]
                box = var[:,:,ind_start_lev:ind_end_lev,ind_start_lat:ind_end_lat,\
                ind_start_lon:ind_end_lon]
        elif lev1<0.:
            if var.ndim==4: #(em,time,lat,lon)
                levs = []
                box = var[:,:,ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
            elif var.ndim==3: #(em,lat,lon)
                levs = []
                box = var[:,ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
    elif em==False:
        if lev1>=0.:
            if lev1!=lev2:
                if var.ndim==5: #(ind,time,lev,lat,lon)
                    ind_start_lev=int(np.abs(lev-(lev1)).argmin())
                    ind_end_lev=int(np.abs(lev-(lev2)).argmin())+1
                    levs = lev[ind_start_lev:ind_end_lev]
                    box = var[:,:,ind_start_lev:ind_end_lev,ind_start_lat:ind_end_lat,\
                    ind_start_lon:ind_end_lon]
                elif var.ndim==4: #(time,lev,lat,lon)
                    ind_start_lev=int(np.abs(lev-(lev1)).argmin())
                    ind_end_lev=int(np.abs(lev-(lev2)).argmin())+1
                    levs = lev[ind_start_lev:ind_end_lev]
                    box = var[:,ind_start_lev:ind_end_lev,ind_start_lat:ind_end_lat,\
                    ind_start_lon:ind_end_lon]
                elif var.ndim==3: #(lev,lat,lon)
                    ind_start_lev=int(np.abs(lev-(lev1)).argmin())
                    ind_end_lev=int(np.abs(lev-(lev2)).argmin())+1
                    levs = lev[ind_start_lev:ind_end_lev]
                    box = var[ind_start_lev:ind_end_lev,ind_start_lat:ind_end_lat,\
                    ind_start_lon:ind_end_lon]
            elif lev1==lev2:
                if var.ndim==4: #(time,lev,lat,lon)
                    ind_lev=int(np.abs(lev-(lev1)).argmin())
                    levs = lev[ind_lev]
                    box = var[:,ind_lev,ind_start_lat:ind_end_lat,\
                    ind_start_lon:ind_end_lon].squeeze()
                elif var.ndim==3: #(lev,lat,lon)
                    ind_lev=int(np.abs(lev-(lev1)).argmin())
                    levs = lev[ind_lev]
                    box = var[ind_lev,ind_start_lat:ind_end_lat,\
                    ind_start_lon:ind_end_lon].squeeze()
        elif lev1<0.:
            if var.ndim==4: #(ind,time,lat,lon)
                levs = []
                box = var[:,:,ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
            elif var.ndim==3: #(time,lat,lon)
                levs = []
                box = var[:,ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
            elif var.ndim==2: #(lat,lon)
                levs = []
                box = var[ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
    return lons,lats,levs,box

def season(var,months=[12,1,2]):
    var_s  =  var.sel(time = np.in1d( var['time.month'], months))
    return var_s

def serial_chunks(var, n_rm, m=12, em=False):
    a = []
    if em==False:
        for i in np.array(range(len(var) - n_rm))[::m]:
            a.append(var[i:i+n_rm])
    elif em==True:
        for i in np.array(range(len(var[0]) - n_rm))[::m]:
            a.append(var[:,i:i+n_rm])
    return a

def reorder_dims(darray, dim1, dim2):
    """
    Interchange two dimensions of a DataArray in a similar way as numpy's swap_axes
    """
    dims = list(darray.dims)
    assert set([dim1,dim2]).issubset(dims), 'dim1 and dim2 must be existing dimensions in darray'
    ind1, ind2 = dims.index(dim1), dims.index(dim2)
    dims[ind2], dims[ind1] = dims[ind1], dims[ind2]
    return darray.transpose(*dims)
