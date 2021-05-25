file=$1
echo ${file}_var_r.nc

#Selvar u,v,pot_temp,rho - select upper 500m from remapped ocean output
cdo -sellevidx,1/27 -selvar,u,v,pot_temp ${file}_var_r.nc tmp_${file}_uvpot_temprho.nc

#Detrend
cdo detrend tmp_${file}_uvpot_temprho.nc tmp_${file}_uvpot_temprho_detrend.nc

#Calculate timmean
cdo timmean tmp_${file}_uvpot_temprho.nc tmp_${file}_uvpot_temprho_timmean.nc

#Add mean to detrended data
cdo -add tmp_${file}_uvpot_temprho_timmean.nc tmp_${file}_uvpot_temprho_detrend.nc ${file}_uvpot_temprho_detrend_mean.nc

#Calculate anomalies
cdo ymonmean ${file}_uvpot_temprho_detrend_mean.nc ${file}_uvpot_temprho_detrend_clim.nc
cdo sub ${file}_uvpot_temprho_detrend_mean.nc ${file}_uvpot_temprho_detrend_clim.nc ${file}_uvpot_temprho_detrend_anom.nc

#Remove tmp files
# rm tmp_${file}_uvpot_temprhovpot_temprho_*.nc
