import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')
import numpy as np
#My modules
from clim_indices import selreg

#================================================================
def find_n_consecutive_greater_than_x(mylist, n, x):
    #check for values that are greater than + 1 standard deviation for 5 consecutive months
    idx = 0
    track = 0
    for index,val in enumerate(mylist):
        if val > x:
            idx += 1
        else:
            if idx >= n:
                yield list(np.arange(track,index,1))
            idx = 0
            track = index + 1

def find_n_consecutive_less_than_x(mylist, n, y):
    #check for values that are lesser than - 1 standard deviation for 5 consecutive months
    idx = 0
    track = 0
    for index,val in enumerate(mylist):
        if val < y:
            idx += 1
        else:
            if idx >= n:
                yield list(np.arange(track,index,1))
            idx = 0
            track = index + 1

def composite_idx(ind, n, x):
    n = n
    x = x
    y = -x

    elnino = list(find_n_consecutive_greater_than_x(ind, n, x))
    idx_elnino = [item for sublist in elnino for item in sublist]

    lanina = list(find_n_consecutive_less_than_x(ind, n, y))
    idx_lanina = [item for sublist in lanina for item in sublist]   #233
    return idx_elnino,idx_lanina

def composite_idx_peak(ind, start, x):
    x = x
    y = -x

    idx_nino = []
    for i in np.arange(start,len(ind)-12,12):
        if (ind[i] > x) and (ind[i+1] > x) and (ind[i+2] > x):
            idx_nino.append([i, i+1, i+2])

    idx_nina = []
    for i in np.arange(start,len(ind)-12,12):
        if (ind[i] < y) and (ind[i+1] < y) and (ind[i+2] < y):
            idx_nina.append([i, i+1, i+2])

    idx_nino = [item for sublist in idx_nino for item in sublist]
    idx_nina = [item for sublist in idx_nina for item in sublist]

    return idx_nino, idx_nina

def find_yrs(time, idx):

    #Years of interest
    yrs = np.unique(time[idx].year)

    if (len(yrs) == 1) and (yrs[0]!=2016):
        all_yrs = [yrs[0], yrs[0]+1]
    elif (len(yrs) == 1) and (yrs[0]==2016):
        all_yrs = []
    elif (len(yrs) == 1) and (yrs[0]==1951):
        all_yrs = []
    else:

        #Select residual yrs of interest
        diff = yrs[1:] - yrs[:-1]

        res_yrs = []
        list_iter = iter(range(len(diff)))
        for i in list_iter:
            res_yrs.append(yrs[i]+1)
            if diff[i] == 1:
                next(list_iter, None)
                continue

        #Handle last year
        if (diff[-1] > 1) and (yrs[-1] != 2016):
            res_yrs = np.append(res_yrs, yrs[-1]+1)
        else:
            pass
        if (diff[-1] > 1) and (yrs[-1] == 2016):
            yrs = np.delete(yrs,np.where(yrs==2016))
        else:
            pass

        #Concatenate years
        all_yrs = np.sort(np.unique(np.concatenate((res_yrs, yrs))))

        if (diff[-1] == 1) and (all_yrs[-1] != 2016) and (len(all_yrs)%2!=0):
            all_yrs = np.append(all_yrs, yrs[-1]+1)
        else:
            pass

    return all_yrs

def find_yrs_peak(time, idx):
    yrs = np.unique(time[idx])
    yrs0 = yrs
    # yrs_1 = yrs0-1
    yrs1 = yrs0+1
    all_yrs = np.sort(np.concatenate((yrs0, yrs1)))
    return all_yrs

def calc_num_events(all_yrs):
    num = int(len(all_yrs)/2)
    return num

def calc_idx_yr01(time_yr, all_yrs):
    # yr_1 = all_yrs[::3]
    # idx_yr_1 = np.where(np.in1d(time_yr,yr_1))[0]
    yr0 = all_yrs[::2]
    idx_yr0 = np.where(np.in1d(time_yr,yr0))[0]
    yr1 = all_yrs[1::2]
    idx_yr1 = np.where(np.in1d(time_yr,yr1))[0]
    return idx_yr0, idx_yr1

def reshape_to_year_mon(var, em=True):
    if em==True:
        if var.ndim == 3:
            var_r = np.stack(np.split(var, var.shape[0]//12, axis=0), axis=0)
        elif var.ndim == 4:
            var_r = np.stack(np.split(var, var.shape[1]//12, axis=1), axis=1)
        elif var.ndim == 5:
            var_r = np.stack(np.split(var, var.shape[1]//12, axis=1), axis=1)
    elif em==False:
        var_r = np.stack(np.split(var, var.shape[0]//12, axis=0), axis=0)
    return var_r

def calc_var_yr01(var_r, idx_yr0, idx_yr1):
    if var_r.ndim==4:
        # var_yr_1 = np.take_along_axis(var_r,idx_yr_1[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
        var_yr0 = np.take_along_axis(var_r,idx_yr0[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
        var_yr1 = np.take_along_axis(var_r,idx_yr1[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
        #concatenate year 0 and year1 months
        var_yr_101 = np.concatenate((var_yr0,var_yr1),axis=1)
    elif var_r.ndim==5:
        # var_yr_1 = np.take_along_axis(var_r,idx_yr_1[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],axis=0)
        var_yr0 = np.take_along_axis(var_r,idx_yr0[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],axis=0)
        var_yr1 = np.take_along_axis(var_r,idx_yr1[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],axis=0)
        # concatenate year 0 and year1 months
        var_yr_101 = np.concatenate((var_yr0,var_yr1),axis=1)
    return var_yr_101

def composite_var(var, ie_yr0, ie_yr1, il_yr0, il_yr1, lon, lat, lev, \
                  lon1, lon2, lat1, lat2, lev1=-999., lev2=-999., em=True, \
                  latmean=True, lonmean=False, levmean=False):
    #reshape variables to 4D
    var_r = reshape_to_year_mon(var, em=em)

    if em==True:
        #Warm events
        evar = np.vstack([calc_var_yr01(var_r[i], ie_yr0[i], ie_yr1[i]) \
        for i in range(len(var_r))])
        #Cold events
        lvar = np.vstack([calc_var_yr01(var_r[i], il_yr0[i], il_yr1[i]) \
        for i in range(len(var_r))])
    elif em==False:
        #Warm events
        evar = calc_var_yr01(var_r, ie_yr0, ie_yr1)
        #Cold events
        lvar = calc_var_yr01(var_r, il_yr0, il_yr1)

    #Select region
    lont, latt, levt, evar = selreg(evar, lon, lat, lev, lon1, lon2, lat1, lat2, \
                                   lev1=lev1, lev2=lev2, em=em)
    lont, latt, levt, lvar = selreg(lvar, lon, lat, lev, lon1, lon2, lat1, lat2, \
                                   lev1=lev1, lev2=lev2, em=em)
    if latmean==True:
        evar = np.nanmean(evar, axis=-2)
        lvar = np.nanmean(lvar, axis=-2)
    else:
        pass

    if lonmean==True:
        evar = np.nanmean(evar, axis=-1)
        lvar = np.nanmean(lvar, axis=-1)
    else:
        pass

    if levmean==True:
        evar = np.nanmean(evar, axis=2)
        lvar = np.nanmean(lvar, axis=2)
    else:
        pass

    return lont, evar, lvar

def calc_ensmean(evar, lvar):
    evar = np.nanmean(evar, axis=0)
    lvar = np.nanmean(lvar, axis=0)
    return evar, lvar

def count_warm_cold(ind_seas, x=1.):
    x = x
    y = -x

    widx = []
    for i in np.arange(len(ind_seas)-1):
        if (ind_seas[i] > x):
            widx.append([i])
    count_warm = len(widx)

    cidx = []
    for i in np.arange(len(ind_seas)-1):
        if (ind_seas[i] < y):
            cidx.append([i])
    count_cold = len(cidx)

    widx = [item for sublist in widx for item in sublist]
    cidx = [item for sublist in cidx for item in sublist]

    return count_warm, count_cold, widx, cidx
