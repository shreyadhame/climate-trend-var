; ****************************************************
load "/srv/ccrc/data25/z5166746/Scripts/my_scripts/RWSa.nl"
load "/srv/ccrc/data25/z5166746/Scripts/my_scripts/WaveAF3.nl"
; ****************************************************

begin

; *********************
; Reading Data
; *********************
nexp = nexp
season = season
n = n
;n = 65

path = "/srv/ccrc/data25/z5166746/IOtrendX_pm/atm/monthly/raw_data/"+nexp+"/rws_files/"

anom_data = addfile(path + nexp + ".pa_1951-2016_var.nc_" + season + "_anom.nc", "r")
clim_data = addfile(path + nexp + ".pa_1951-2016_var.nc_" + season + "_clim.nc", "r")

do i=0,n

  uc = clim_data->ua_plev(:,{200},:,:)
  uc@missing_value= -999
  uc@_FillValue = -999
  vc = clim_data->va_plev(:,{200},:,:)
  vc@missing_value= -999
  vc@_FillValue = -999
  ua = anom_data->ua_plev(i,{200},:,:)
  ua@missing_value= -999
  ua@_FillValue = -999
  va = anom_data->va_plev(i,{200},:,:)
  va@missing_value= -999
  va@_FillValue = -999
  za = anom_data->zg_plev(i,{200},:,:)
  za@missing_value= -999
  za@_FillValue = -999
  level = 200.

  lat = anom_data->latitude_0
  lon = anom_data->longitude_0
  nlat = dimsizes(lat)
  nlon = dimsizes(lon)

  ucm = dim_avg_n_Wrap(uc, 0)
  vcm = dim_avg_n_Wrap(vc, 0)
  uam = dim_avg_n_Wrap(ua, 0)
  vam = dim_avg_n_Wrap(va, 0)
  zam = dim_avg_n_Wrap(za, 0)

  ; **************************************************
  ; Calculating RWS
  ; **************************************************

  ;RWS  = RWSa(uam, vam, ucm, vcm)
  RWS  = RWSa(ua, va, ucm, vcm)
  ud   = RWS[0]           ; Divergent wind anomaly, zonal component
  vd   = RWS[1]           ; Divergent wind anomaly, meridional component
  S1  = RWS[2]         ; Advection of abs vort by anom divergent wind
  S2  = RWS[3]         ; Vortex stretching
  S   = S1 + S2           ; Total source

  ;WAF = WaveAF3(zam, ucm, vcm, level, lat, lon)
  WAF = WaveAF3(za, ucm, vcm, level, lat, lon)
  fx = WAF[0]
  fy = WAF[1]
  psidev = WAF[2]

  ; ***********************************************
  ; Saving variable into netcdf file
  ; ***********************************************

  setfileoption("nc","format","netcdf4")

  diro = path                   ; Output directory
  filo = nexp + "_rws_waf_" + season + "_" + sprinti("%0.2i", i) + ".nc"
  system("/bin/rm -f " + diro + filo)    ; remove if exists
  ncdf    = addfile(diro + filo, "c")

  ; make time and UNLIMITED dimension      ; recommended  for most applications
  filedimdef(ncdf,"time",-1,True)

  ; output variables directly

  S!0    = "lat"                      ; assign named dimensions
  S!1    = "lon"
  S&lat  =  lat
  S&lon  =  lon

  S@long_name = "Rossby wave source"
  S@units = "/s"
  ncdf->S = S

  S1@long_name = "Advection of absolute vorticity by anomalous divergent wind"
  S1@units = "/s"
  ncdf->S1 = S1

  S2@long_name = "Vortex stretching"
  S2@units = "/s"
  ncdf->S2 = S2

  ud@long_name = "u-divergent wind"
  ud@units = "m/s"
  ncdf->ud = ud

  vd@long_name = "v-divergent wind"
  vd@units = "m/s"
  ncdf->vd = vd

  fx@long_name = "x-component of wave activity flux"
  fx@units = "m2/s2"
  ncdf->fx = fx

  fy@long_name = "y-component of wave activity flux"
  fy@units = "m2/s2"
  ncdf->fy = fy

  psidev@long_name = "quasi geostrophic stream function"
  psidev@units = "m2/s"
  ncdf->psidev = psidev

end do
end
exit
