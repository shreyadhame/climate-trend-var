#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create netcdf file

"""

__title__ = "Create a netcdf file for IO trends and global climatology"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#===================================================
#General modules
import argparse
import netCDF4 as nc4
import numpy as np
import numpy.ma as ma
import xarray as xr
from datetime import datetime
from gridfill import fill
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmocean import cm as cmo
import xarray as xr
from datetime import datetime

#My modules
from read import read_data, seltime
from clim_indices import clim_anom_xr
from trend import *

#===============================================================

def mask_oceans(mask_file, var, lon, lat, o=[3,56]):
    #read mask file
    with open (mask_file,'r') as f:
        lines = f.readlines()
    mask_data = lines[2:]
    if var.ndim ==3:
        maskdata=np.zeros((var.shape[1],var.shape[2]))
    elif var.ndim ==2:
        maskdata=np.zeros((var.shape[0],var.shape[1]))
    for line in mask_data:
        line = line.strip()
        columns = line.split(",")
        lat_m = columns[0]
        lon_m = float(columns[1])%360
        basin = columns[2]
        #Select basin number that is NOT to be masked
        if any(int(basin)== v for v in o):
            if len(np.where(lon == float(lon_m))[0]) > 0:
                maskdata[np.where(lat == float(lat_m))[0][0],np.where(lon == float(lon_m))[0][0]] = 1.
            if len(np.where(lon == float(lon_m)-0.5)) > 0:
                maskdata[np.where(lat == float(lat_m)-0.5),np.where(lon == float(lon_m)-0.5)] = 1.
    f.close()
    if var.ndim ==3:
        maskdata = np.repeat(maskdata[np.newaxis,:,:], var.shape[0], axis=0)
        var_mask = var.where(maskdata ==1.)
    elif var.ndim ==2:
        var_mask = var.where(maskdata ==1.)
    return var_mask

def damp(trend_io,lon,lat):
    #Vertical damping from -35.5 to -49.5 lat
    #Find latlon indices
    ind_start_lat=int((np.abs(lat-(-35.5))).argmin()) #start damping from here
    ind_end_lat=int((np.abs(lat-(-49.5))).argmin()) #last lat with trend data
    ind_start_lon=int((np.abs(lon-(20.5))).argmin()) #IO lon1
    ind_end_lon=int((np.abs(lon-(146.5))).argmin()) #IO lon2
    #Select latlon box to modify
    to_modify = trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]

    dmp = np.zeros(to_modify.shape)
    c = dmp.shape[0] #number of decrements
    for i in range(dmp.shape[1]):
        dmp[:,i] = np.linspace(1,0.,c)
    modified = np.multiply(to_modify,dmp)
    #replace modifed boxes in original dataset
    trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]=modified #dmp_flip

    #Horizontal damping from 15.5 to 20.5 longitude
    #Find latlon indices
    ind_start_lat=int((np.abs(lat-(-35.5))).argmin()) #start damping from here
    ind_end_lat=int((np.abs(lat-(-49.5))).argmin()) #last lat with trend data
    ind_start_lon=int((np.abs(lon-(15.5))).argmin()) #IO lon1
    ind_end_lon=int((np.abs(lon-(20.5))).argmin()) #IO lon2
    #Select latlon box to modify
    to_modify = trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]

    dmp = np.zeros(to_modify.shape)
    c = dmp.shape[1] #number of decrements
    for i in range(dmp.shape[0]):
        dmp[i,:] = np.flip(np.linspace(to_modify[i,-1],0.,c),axis=0)
    #replace modifed boxes in original dataset
    trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]=dmp #dmp_flip

    #Horizontal damping from 141.5 to 146.5 longitude
    #Find latlon indices
    ind_start_lat=int((np.abs(lat-(-35.5))).argmin()) #start damping from here
    ind_end_lat=int((np.abs(lat-(-49.5))).argmin()) #last lat with trend data lat with trend data
    ind_start_lon=int((np.abs(lon-(146.5))).argmin()) #IO lon1
    ind_end_lon=int((np.abs(lon-(151.5))).argmin()) #IO lon2
    #Select latlon box to modify
    to_modify = trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]

    # #Interpolate masked values
    kw = dict(eps=1e-4, relax=0.6, itermax=1e4, initzonal=False,
          cyclic=False, verbose=True)
    to_modify_int, converged = fill(np.ma.masked_invalid(to_modify), 1, 0, **kw)

    dmp = np.zeros(to_modify.shape)
    c = dmp.shape[1]
    for i in range(dmp.shape[0]):
        dmp[i,:] = np.linspace(to_modify_int[i,0],0.,c)
    #replace modifed boxes in original dataset
    trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]=dmp #dmp_flip

    #Damping near ITF
    #Find latlon indices
    ind_start_lat=int((np.abs(lat-(20.5))).argmin())
    ind_end_lat=int((np.abs(lat-(-20.5))).argmin())
    ind_start_lon=int((np.abs(lon-(91.5))).argmin())
    ind_end_lon=int((np.abs(lon-(138.5))).argmin())
    #Select latlon box to modify
    to_modify = trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]

    #Find the coastline
    endpoints = np.zeros(to_modify.shape[0],dtype=int)
    for i in range(to_modify.shape[0]):
        endpoints[i] = np.max(np.where(to_modify[i]>0))

    #Find all end points to damp from
    lists = [[] for i in range(len(endpoints))]
    for i in range(len(endpoints)):
        if i==0:
            lists[i] = [2]
        elif i!=0:
            if endpoints[i] > endpoints[i-1]:
                lists[i] = (np.arange(endpoints[i-1],endpoints[i]+1,1))
            else:
                lists[i] = [endpoints[i]]

    #damp along diagonals
    for i in range(len(lists)):
        for j in range(len(lists[i])):
            b = i
            if i > 7:
                a = b-7
            else:
                a = 0
            c = lists[i][j]
            d = c + (b-a)
            box = np.nan_to_num(to_modify[a:b+1,c:d+1])
            box_flip = box[::-1]
            dmp = np.linspace(box_flip[0,0],0.,len(box_flip[0]))
            np.fill_diagonal(box_flip,dmp)
            box_back = box_flip[::-1]
            to_modify[a:b+1,c:d+1] = box_back

    #mask zeros
    to_modify = ma.masked_where(to_modify == 0., to_modify)

    #interpolate values
    kw = dict(eps=1e-4, relax=0.6, itermax=1e4, initzonal=False,
          cyclic=False, verbose=True)
    to_modify_int, converged = fill(np.ma.masked_invalid(to_modify), 1, 0, **kw)

    #replace modified box in original dataset
    trend_io[ind_start_lat:ind_end_lat+1,ind_start_lon:ind_end_lon+1]=to_modify_int

    return trend_io

def create_netcdf(fname,idate,idatesec,ilon,ilat,itime,iice,isst):
    #Write a new netcdf file
    f = nc4.Dataset(fname,'w', format='NETCDF4')
    #Specify dimensions
    f.createDimension('lon',len(ilon))
    f.createDimension('lat',len(ilat))
    f.createDimension('time', None)
    #Build Variables
    date = f.createVariable('date','int','time')
    datesec = f.createVariable('datesec','int','time')
    lon = f.createVariable('lon', 'double', 'lon')
    lat = f.createVariable('lat', 'double', 'lat')
    time = f.createVariable('time', 'double', 'time')
    ice_cov = f.createVariable('ice_cov', 'float', ('time', 'lat', 'lon'))
    SST_cpl = f.createVariable('SST_cpl', 'float', ('time', 'lat', 'lon'))
    ice_cov_prediddle = f.createVariable('ice_cov_prediddle', 'float', ('time', 'lat', 'lon'))
    SST_cpl_prediddle = f.createVariable('SST_cpl_prediddle', 'float', ('time', 'lat', 'lon'))
    #Add attributes
    f.description = "Boundary Condition Data for IO SST trend"
    f.history = "Created" + datetime.today().strftime("%d/%m/%y")
    date.long_name = "current date (YYYYMMDD)"
    datesec.long_name = "current seconds of current date"
    lon.long_name = "longitude"
    lon.units = "degrees east"
    lat.long_name = "latitude"
    lat.units = "degrees north"
    time.units = "days since 1950-01-01 00:00:00"
    time.calendar = "365_day"
    ice_cov.long_name = "BCS Pseudo Sea-ice concentration"
    ice_cov.units = "fraction"
    SST_cpl.long_name = "BCS Pseudo SST"
    SST_cpl.units = "deg_C"
    ice_cov_prediddle.long_name = "Sea-ice concentration before time diddling"
    ice_cov_prediddle.units = "fraction"
    SST_cpl.long_name = "BCS Pseudo SST"
    SST_cpl.units = "deg_C"
    SST_cpl_prediddle.long_name = "SST before time diddling"
    SST_cpl_prediddle.units = "deg_C"
    #Pass data into Variables
    date[:] = idate
    datesec[:] = idatesec
    lon[:] = ilon
    lat[:] = ilat
    time[:] = itime
    ice_cov[:,:,:] = iice
    SST_cpl[:,:,:] = isst
    ice_cov_prediddle[:,:,:] = iice
    SST_cpl_prediddle[:,:,:] = isst

    #close Dataset
    f.close()

#===============================================================

### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ivar_obs')
    parser.add_argument('time_ifile')
    parser.add_argument('fname')
    args = parser.parse_args()

    #Extract lon,lat,sst,time from HadISST observation (1950-2017)
    ipath_had = '/srv/ccrc/data25/z5166746/Obs_data/sst/HadISST/HadISST_all_clean.nc'
    lon_had,lat_had,lev_had,sst_had,time_had,basin_mask = read_data(ipath_had,'sst',\
    imask=None)

    #Select time
    start_time = 1950
    end_time = 2017
    sst_t,time_t = seltime(sst_had,time_had,start_time,end_time)

    # #Calculate area weighed ssts
    # wgtfac = areawgtvar_3D(lon_had,lat_had)
    # sst_had_aw = np.multiply(sst_had,wgtfac[np.newaxis,...])
    #
    #Mask other oceans
    sst_io = mask_oceans('./../../grids/basinmask_01.msk', sst_had, lon_had, lat_had)

    #Calculate trend in IO SST from 1950 to 2017
    trend_io = calc_func(sst_io,mk_test,ncores=ncores,noutput=noutput)[6]

    # #   1   Assign mean trend to all grid cells
    # trend_io_m = xr.full_like(trend_io,np.mean(trend_io))
    # trend_io_m = mask_oceans(args.imask, trend_io_m, lon, lat)

    # #   2   Assign double the mean trend
    # trend_io_2m = xr.full_like(trend_io,2*np.mean(trend_io))
    # trend_io_2m = mask_oceans(args.imask, trend_io_2m, lon, lat)

    #  3   Assign double the mean trend
    #trend_io_3m = xr.full_like(trend_io,3*np.mean(trend_io))
    #trend_io_3m = mask_oceans(args.imask, trend_io_3m, lon, lat)

    # Damp the edges
    trend_io_dmp = damp(trend_io,lon,lat)
    #trend_io_dmp = trend_io_dmp.values

    #Calculate global climatology
    clim = clim_anom_xr(sst_t,time_t,start=None)[0]
    clim = clim.values
    #clip T
    clim[clim < -1.77] = -1.77
    #Interpolate sst
    clim_int = np.zeros(clim.shape)
    for i in range(clim.shape[0]):
        clim_int[i], converged = fill(ma.masked_invalid(clim[i]), 1, 0, **kw)

    #Repeat the climatology 68 times (1950-2017)
    nyears =(end_time - start_time)+1
    clim_rpt = np.repeat(clim_int[np.newaxis,:,:,:],nyears,axis=0)\
    .reshape(clim_int.shape[0]*nyears,clim_int.shape[1],clim_int.shape[2])

    # Repeat IO trend 68 times (1950-1975)
    trend_io_rpt = np.repeat(trend_io_int[np.newaxis,:,:],nyears*12,axis=0)
    # Create an array to multiply rates with
    multiplier = np.arange(1,(nyears*12)+1,1)[:,np.newaxis,np.newaxis]
    # Multiply rate with multiplier to get transient trends
    trend_io_trans = np.multiply(trend_io_rpt,multiplier)

    #Combine climatology with trends
    clim_trend = clim_rpt + trend_io_trans

    #Input SST
    ssti = clim_trend

    #Extract sea ice extent from Hadley data
    #Extract sea ice extent from Hadley data
    lon_sic,lat_sic,lev_sic,sic,time_sic,basin_mask = read_data( '/srv/ccrc/data25/z5166746/Obs_data/sst/HadISST/HadISST_ice.nc','sic',\
    imask=None)
    #Select time period
    sic_t,time_t = seltime(sic,time_sic,start_time,end_time)
    #Calculate global SIC climatology
    clim_sic = clim_anom_xr(sic_t,time_t,start=None)[0]
    clim_sic = clim_sic.values

    #Interpolate ice
    clim_sic_int = np.zeros(clim_sic.shape)
    for i in range(clim_sic.shape[0]):
        clim_sic_int[i], converged = fill(ma.masked_invalid(clim_sic[i]), 1, 0, **kw)

    #Repeat the climatology 68 times (1950-2017)
    sici = np.repeat(clim_sic_int[np.newaxis,:,:,:],nyears,axis=0)\
    .reshape(clim_sic_int.shape[0]*nyears,clim_sic_int.shape[1],clim_sic_int.shape[2])

    #Flip lat and data along lat axis
    lat = np.flip(lat,axis=0)
    sici = np.flip(sici,axis=1)
    ssti = np.flip(ssti,axis=1)

    #Create time variables
    #date
    years = np.arange(19500000,20180000,10000)
    years = np.repeat(years,12)
    months = np.arange(100,1300,100)
    months = np.tile(months,nyears)
    days = np.array([16,15,16,16,16,16,16,16,16,16,16,16])
    days = np.tile(days,nyears)
    idate = years + months + days

    #datesec
    datesec = np.array([43200,0,43200,0,43200,0,43200,43200,0,43200,0,43200])
    idatesec = np.tile(datesec,nyears)

    #time
    #Extract time from CESM inputdata
    ds = xr.open_dataset(args.time_ifile,decode_times=False)
    time = ds.time
    itime = time[:nyears*12]

    # #Extract date, datesec, time from CESM inputdata for climatological file
    # ds = xr.open_dataset(args.inputfile,decode_times=False)
    # idate = ds.date
    # idatesec = ds.datesec
    # itime = ds.time

    create_netcdf(args.fname,idate,idatesec,lon,lat,itime,sici,ssti)
