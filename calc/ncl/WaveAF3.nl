;Guomin's code
; Adapted for my use (changed calendar stuff)
;===============================================================
undef ("WaveAF3")
function WaveAF3(zavar,cuvar,cvvar,level,lat,lon)
;Input
; zavar(y,x) - z anom (m). mn either time or mode (geopotential height)
; cuvar,cvvar - basic wind, same dimentionality as zavar (m/s)
; level - data p level (mb), scalar
; lat,lon - data coordinates (deg)
;Return
; Fx,Fy - x,y components of wave activity fluxes
; psidev - geostrophic streamfuction derived from zavar
local uc,dc,nlat,nlon,gc,ga,re,sclhgt,pi,f,coslat,leveltmp \
     ,coslattmp,ftmp,cumag,tumag,tuvar,tvvar,psidev,dpsidevdlon \
     ,ddpsidevdlonlon,dpsidevdlat,ddpsidevdlonlat,ddpsidevdlatlat \
     ,xuterm,xvterm,yvterm,Fx,Fy
begin

; primary control parameters
uc = 2 ; mean zonal wind less than uc masked. [uc = 5 m/s was used]
dc = 5 ; mask deep tropics |lat|<dc deg. Was 10.

nlat = dimsizes(lat)
nlon = dimsizes(lon)

;  Gas constant
gc=290
;  Gravitational acceleration
ga=9.80665
;  Radius of the earth
re=6378388
; scale height
sclhgt=8000.
; pi
pi = atan(1.0)*4.
; Coriolis parameter
f =  2.*2.*pi/(60.*60.*24.)*sin(pi/180. * lat(:))
f!0 = "lat"
f&lat = lat
f@_FillValue = cuvar@_FillValue

; missing deep tropics (was 10N-10S)
do ilat = 0, nlat-1
 if (abs(lat(ilat) ).lt. dc ) then
  f(ilat)= f@_FillValue
 end if
end do
;cosine
coslat = cos(lat(:)*pi/180.)

; 1-D -> 3-D
leveltmp = conform_dims(dimsizes(zavar),level,-1) ; from scalar
coslattmp = conform_dims(dimsizes(zavar),coslat,0)
ftmp = conform_dims(dimsizes(zavar),f,0)

; magnitude of climatological wind
cumag = sqrt(cuvar^2 + cvvar^2)
cumag@_FillValue = cuvar@_FillValue
cumag = where(cumag .gt. 0, cumag, cumag@_FillValue) ; [z,y,x]


copy_VarMeta(zavar,cumag)
cumag = 2.*cumag*re*re  ; ready form in equation
leveltmp = leveltmp/1000. ; normalized
leveltmp = leveltmp/cumag ; ready form in equation
;printVarSummary(cumag)

; QG stream function for anomaly
psidev = zavar*ga /ftmp  ; [nm,y,x]

; remove zonal mean
psiza = dim_avg_n_Wrap(psidev,1)  ; zonal average [nm,y]
do ix = 0, nlon-1
  psidev(:,ix) = psidev(:,ix) - psiza
end do

;dpsidev/dlon
dpsidevdlon =  center_finite_diff_n(psidev,lon*pi/180.,True,0,1) ; x dim = 1
;ddpsidev/dlonlon
ddpsidevdlonlon =  center_finite_diff_n(dpsidevdlon,lon*pi/180.,True,0,1)

;dpsidev/dlat
dpsidevdlat = center_finite_diff_n(psidev, lat*pi/180., False,0,0) ; y dim = 0

;ddpsidev/dlonlat
ddpsidevdlonlat =  center_finite_diff_n(dpsidevdlon,lat*pi/180.,False,0,0)

;ddpsidev/dlatlat
ddpsidevdlatlat = center_finite_diff_n(dpsidevdlat, lat*pi/180.,False,0,0)

xuterm = (dpsidevdlon*dpsidevdlon - psidev*ddpsidevdlonlon)
xvterm = (dpsidevdlon*dpsidevdlat - psidev*ddpsidevdlonlat)
;yuterm = xvterm
yvterm = (dpsidevdlat*dpsidevdlat - psidev*ddpsidevdlatlat)

; Mask out where westerlies is small or negative (less than uc m/s). uc was 5
;  by using mask
;x-component of (38)
Fx = mask(leveltmp*(cuvar/coslattmp*xuterm + cvvar*xvterm),cuvar.lt.uc,False)  ; protect where condition is met
;y-component
Fy = mask(leveltmp*(cuvar*xvterm + coslattmp*cvvar*yvterm),cuvar.lt.uc,False)

delete(cumag)
delete(psiza)
delete(dpsidevdlon)
delete(dpsidevdlat)
delete(ddpsidevdlonlon)
delete(ddpsidevdlonlat)
delete(ddpsidevdlatlat)
copy_VarMeta(zavar,Fx)
copy_VarMeta(zavar,Fy)
copy_VarMeta(zavar,psidev)
return ( [/ Fx, Fy, psidev /] )
end
