file=$1 #a10_Mclm_r01.mocn_1951-2016

# echo ${file}_w.nc

#Sign for vertical velocity in ocean is reversed positive is down
# Multiply the file by -1
cdo -mulc,-1. ${file}_w.nc ${file}_w_r.nc

#Detrend data
cdo -detrend ${file}_w_r.nc tmp_${file}_w_r_detrend.nc

#Calculate timmean
cdo timmean ${file}_w_r.nc tmp_${file}_w_r_timmean.nc

#Add mean to detrended data
cdo add tmp_${file}_w_r_detrend.nc tmp_${file}_w_r_timmean.nc ${file}_w_r_detrend_mean.nc

#Calculate climatology
cdo ymonmean ${file}_w_r_detrend_mean.nc ${file}_w_r_detrend_clim.nc

#Calculate anomaly
cdo sub ${file}_w_r_detrend_mean.nc ${file}_w_r_detrend_clim.nc ${file}_w_r_detrend_anom.nc

#Remove tmp files
rm tmp_${file}_w_r_*.nc
rm ${file}_w.nc
