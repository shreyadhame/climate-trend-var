file=$1
echo ${file}_u_r.nc

#Subtract 273.15 from pot_temp
cdo subc,273.15 ${file}_pot_temp_r.nc ${file}_pot_temp_rs.nc
# Selvar u,v,pot_temp,rho - select upper 500m from remapped ocean output
cdo -sellevidx,1/27 -selvar,uo ${file}_u_r.nc tmp_${file}_u.nc
cdo -sellevidx,1/27 -selvar,vo ${file}_v_r.nc tmp_${file}_v.nc
cdo -sellevidx,1/27 -selvar,thetao ${file}_pot_temp_rs.nc tmp_${file}_pot_temp.nc
cdo -sellevidx,1/27 -selvar,rhopoto ${file}_rho_r.nc tmp_${file}_rho.nc
cdo merge tmp_${file}_u.nc tmp_${file}_v.nc tmp_${file}_pot_temp.nc tmp_${file}_rho.nc tmp_${file}_uvpot_temprho.nc

#Detrend
cdo detrend tmp_${file}_uvpot_temprho.nc tmp_${file}_uvpot_temprho_detrend.nc

#Calculate timmean
cdo timmean tmp_${file}_uvpot_temprho.nc tmp_${file}_uvpot_temprho_timmean.nc

#Add mean to detrended data
cdo -add tmp_${file}_uvpot_temprho_timmean.nc tmp_${file}_uvpot_temprho_detrend.nc ${file}_uvpot_temprho_detrend_mean.nc

#Calculate anomalies
cdo ymonmean ${file}_uvpot_temprho_detrend_mean.nc ${file}_uvpot_temprho_detrend_clim.nc
cdo sub ${file}_uvpot_temprho_detrend_mean.nc ${file}_uvpot_temprho_detrend_clim.nc ${file}_uvpot_temprho_detrend_anom.nc
# Remove tmp files
# rm tmp_${file}_uvpot_temprhovpot_temprho_*.nc
