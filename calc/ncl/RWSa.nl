; 29/5 Adapted fromGuomin Wang's code

; Requires u and v files with time, lat, lon
; Designed to be able to acept ERAI, POAMA, etc

; The only things this script really needs is u and v
; The anomalies, divergence and vorticity are calculated from this
; (this also means that the u and v files need time, lat, lon)

;===============================================================
undef ("RWSa")
function RWSa(ua, va, uc, vc)                                   ;RWSa(ym1,ym2,yc1,yc2)
;In
; ua, va zonal and meridional wind anomaly (time, lat, lon)     ;ym1,ym2 - date start/end yr/mon, eg 197901 201612
; uc, vc zonal and meridional wind climatology (time, lat, lon)

; yc1,yc2 - clim yr ranges eg 1981,2010
; this example shows RWS for Sep-Oct mean anomaly. Data path hard-wired.
;OUT
; ud,vd - divergent wind on the same p level as input
; S1,S2 - RWS two components
local lat,lon,timeu,udate,usub,timev,vsub,lu,lv,uc,vc,ti0,pi,f,ftmp \
     ,vrdv,avort,avort3,dv,uvd,nyr,S1,S2

begin

; monthly mean data CHANGE HERE
;lat = ua&lat
;lon = ua&lon

data1 = addfile("/srv/ccrc/data25/z5166746/IOtrendX_pm/atm/monthly/raw_data/a10_Mclm_r01/a10_Mclm_r01.pa_1951-2016_var.nc_ncl","r")
lat = data1->latitude_0
lon = data1->longitude_0

; f & conform f
pi    = atan(1.0)*4.
; Coriolis parameter
f     =  2.*2.*pi/(60.*60.*24.)*sin(pi/180. * lat(:))
f!0   = "lat"
f&lat = lat
ftmp  = conform_dims(dimsizes(uc),f,0) ; f to 2D [:,:]
copy_VarMeta(uc,ftmp)

;------ clim absolute vorticity --------
; uv2vrdvF: NCL function calculates div [1] and vort [0]
vrdv  = uv2vrdvF (uc,vc)   ;u,v ==> (2,:,:) 0 relative vort 1 div
avort = vrdv(0,:,:) + ftmp ; absolute vorticity [:,:]
copy_VarMeta(uc,avort)

; anom divergence & divergence wind
vrdv := uv2vrdvF (ua,va)   ; ua,va ==> (2,:,:) 0 relative vort 1 div
dv    = vrdv(1,:,:)        ; divergence anom        ;dv2uvF: NCL function gets irrotational component
uvd   = dv2uvF (dv)        ; dv  ==> divergent wind anom; [0] is u component, [1] is v component
ud    = uvd(0,:,:)         ; u component of divergent wind anomaly
vd    = uvd(1,:,:)         ; v component of divergent wind anomaly
copy_VarMeta(ua,dv)
copy_VarMeta(ua,ud)
copy_VarMeta(ua,vd)

; RWS terms dv & S2 both smoothed
copy_VarMeta(ud,avort)  ; u must be south to north, avort is scalar (required for advect_variable)?

S1 = advect_variable(ud,vd,avort,1," "," ",0)  ; 1 fixed grids 0 return adv
S1 = -S1    ; adjust sign
 dv = smth9_Wrap(dv,0.50,0.25,True) ; heavy smoothing; cyclic
S2 = -avort*dv
S2 = smth9_Wrap(S2,0.50,0.25,True) ; heavy smoothing; cyclic
copy_VarMeta(S1,S2)
;printVarSummary(ud)
;printVarSummary(vd)
;printVarSummary(S1)
;printVarSummary(S2)

;delete(vrdv)
;delete(dv)
;delete(ua)
;delete(va)
;delete(uc)
;delete(vc)
;delete(uvd)
;delete(avort)

return ( [/ ud,vd,S1,S2 /] )
end
