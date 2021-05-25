file=$1
echo ${file}_uvpot_temp.nc

#Detrend
cdo detrend ${file}_uvpot_temp.nc tmp_${file}_uvpot_temp_detrend.nc

#Calculate timmean
cdo timmean ${file}_uvpot_temp.nc tmp_${file}_uvpot_temp_timmean.nc

#Add mean to detrended data
cdo -add tmp_${file}_uvpot_temp_timmean.nc tmp_${file}_uvpot_temp_detrend.nc ${file}_uvpot_temp_detrend_mean.nc

#Calculate anomalies
cdo ymonmean ${file}_uvpot_temp_detrend_mean.nc ${file}_uvpot_temp_detrend_clim.nc
cdo sub ${file}_uvpot_temp_detrend_mean.nc ${file}_uvpot_temp_detrend_clim.nc ${file}_uvpot_temp_detrend_anom.nc

# Remove tmp files
rm tmp_${file}_uvpot_temp*.nc
