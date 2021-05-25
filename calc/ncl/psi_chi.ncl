; ****************************************************
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
 load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
; ****************************************************

begin

; *********************
; Reading Data
; *********************
; ACCESS
;nexp = nexp
;path = "/srv/ccrc/data25/z5166746/IOtrendX_pm/atm/monthly/raw_data/"+nexp+"/"
;data1 = addfile(path + nexp+".pa_1951-2016_var.nc_ncl","r")

;lat = data1->latitude_0
;lon = data1->longitude_0
;lev = data1->pressure ;hPa
;lev = lev(::-1)*100 ; convert units and arrange top to bottom
;lev@units = "Pa"
;time = data1->time
;;xx = dimsizes(lat)
;yy = dimsizes(lon)

;U = data1->ua_plev(:,::-1,:,:)    ;m/s;(:,{100},:,:)
;V = data1->va_plev(:,::-1,:,:)    ;m/s;(:,{100},:,:)

;PS = data1->psl ;Pa

;T = data1->ta_plev(:,::-1,:,:) ;K

;Q = data1->hus_plev(:,::-1,:,:)

; *********************
; CAM5
; Control
;nexp = "CTRL"
;path = "/srv/ccrc/data25/z5166746/IOtrendX_atm/Atm_CTRL_obsclim/"
;data1 = addfile(path + "Atm_CTRL.cam.h0.1951-2016.nc","r")

; IOtrend
nx = nx
nexp = nexp
path = "/srv/ccrc/data25/z5166746/IOtrendX_atm/Atm_IOtrend_obsclim/"+nx+"/"+nexp+"/"
data1 = addfile(path + "Atm_IOtrend_"+nexp+".cam.h0.1951-2016.nc", "r")

lat = data1->lat
lon = data1->lon
lev = data1->lev
lev = lev*100 ; convert units
lev@units = "Pa"
time = data1->time
;xx = dimsizes(lat)
;yy = dimsizes(lon)

U = data1->U(:,:,:,:)    ;m/s;(:,{100},:,:)
V = data1->V(:,:,:,:)    ;m/s;(:,{100},:,:)

PS = data1->PSL ;Pa

T = data1->T(:,:,:,:) ;K

Q = data1->Q(:,:,:,:) ;kg/kg

U@missing_value= -999
U@_FillValue = -999
V@missing_value = -999
V@_FillValue = -999

; desired output levels [hPa/mb]
;pnew = (/  10, 20, 30, 50, 70,100,150,200,250 \
;           , 300,400,500,600,700,800,850,900,950,1000 /)

; **************************************************
; Calculating Zonal Mean Meridional Streamfunction
; **************************************************
zmpsi = zonal_mpsi_Wrap(V,lat,lev,PS)

; **************************************************
; Calculating Streamfunction and Velocity Potential
; **************************************************
var_tmp = uv2sfvpF(U,V)  ;/1e6 fixed grid
psi = var_tmp(0,:,:,:,:)    ; Streamfunction (psi)
chi = var_tmp(1,:,:,:,:)   ; Velocity Potential (chi)

; **************************************************
; Calculating Divergence and Surface Moisture Flux Divergence
; **************************************************
dv = uv2dvF(U,V)  ;Divergence
qu = Q(:,{850},:,:) * U(:,{850},:,:)
qv = Q(:,{850},:,:) * V(:,{850},:,:)
mfd = uv2dv_cfd(qu,qv,lat,lon,0)

; **************************************************
; Calculating Potential Vorticity, Static Stability, Potential Temperature
; **************************************************
gridType = 1 ;fixed grid
opt = 1
var_tmp2 = pot_vort_isobaric(lev, U, V, T, lat, gridType, opt) ;Potential Vorticity
pv = var_tmp2[0]
ss = var_tmp2[1]
pt = var_tmp2[2]

; *******************************************
; Checking variable dimensions
; *******************************************
;printVarSummary(dv)

; ***********************************************
; Deleting variables to save memory allocation
; ***********************************************
delete(U)
delete(V)
delete(T)
delete(Q)
delete(var_tmp)
delete(var_tmp2)
delete(data1)

; ***********************************************
; Saving variable into netcdf file
; ***********************************************

setfileoption("nc","format","netcdf4")

diro = path                     ; Output directory
filo = diro + nexp+".ncl_1951-2016.nc"
system("/bin/rm -f " + diro + filo)    ; remove if exists
ncdf    = addfile(filo, "c")

; make time and UNLIMITED dimension      ; recommended  for most applications
filedimdef(ncdf,"time",-1,True)

; output variables directly

zmpsi!0    = "time"
zmpsi!1    = "lev"
zmpsi!2    = "lat"
zmpsi&time =  time
zmpsi&lev  =  lev
zmpsi&lat  =  lat

zmpsi@long_name = "Zonal mean meridional streamfunction"
zmpsi@units = "kg/s"
ncdf->ZMPSI = zmpsi

psi!0    = "time"
psi!1    = "lev"
psi!2    = "lat"                      ; assign named dimensions
psi!3    = "lon"
psi&time =  time
psi&lev  =  lev
psi&lat  =  lat
psi&lon  =  lon

psi@long_name = "Streamfunction"
psi@units = "m2/s"
ncdf->PSI = psi

chi!0    = "time"
chi!1    = "lev"
chi!2    = "lat"
chi!3    = "lon"
chi&time =  time
chi&lev  =  lev
chi&lat  =  lat
chi&lon  =  lon

chi@long_name = "Velocity Potential"
chi@units = "m2/s"
ncdf->CHI = chi

dv!0    = "time"
dv!1    = "lev"
dv!2    = "lat"
dv!3    = "lon"
dv&time =  time
dv&lev  =  lev
dv&lat  =  lat
dv&lon  =  lon

dv@long_name = "Divergence"
dv@units = "/s"
ncdf->DV = dv

mfd!0    = "time"
mfd!1    = "lat"
mfd!2    = "lon"
mfd&time =  time
mfd&lat  =  lat
mfd&lon  =  lon

mfd@long_name = "Moisture flux divergence"
mfd@units = "kg/kg/s"
ncdf->MFD = mfd

pv!0    = "time"
pv!1    = "lev"
pv!2    = "lat"
pv!3    = "lon"
pv&time =  time
pv&lev  =  lev
pv&lat  =  lat
pv&lon  =  lon

pv@long_name = "Potential Vorticity"
pv@units = "K m2/kg/s"
ncdf->PV = pv

ss!0    = "time"
ss!1    = "lev"
ss!2    = "lat"
ss!3    = "lon"
ss&time =  time
ss&lev  =  lev
ss&lat  =  lat
ss&lon  =  lon

ss@long_name = "Static stability"
ss@units = "K/Pa"
ncdf->SS = ss

pt!0    = "time"
pt!1    = "lev"
pt!2    = "lat"
pt!3    = "lon"
pt&time =  time
pt&lev  =  lev
pt&lat  =  lat
pt&lon  =  lon

pt@long_name = "Potential Temperature"
pt@units = "K"
ncdf->PT = pt

end
exit
