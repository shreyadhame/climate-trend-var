#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Calculate wind stress"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#==============================================================================
import numpy as np
#==============================================================================

#Meteorogical constants
kappa = 0.4
charn = Charnock_alpha = 0.011
g = 9.8
R_roughness = 0.11

def cdn(sp, z, Ta=10):
    """Computes neutral drag coefficient.
    Methods available are: Large & Pond (1981),  Vera (1983) or Smith (1988)
    Parameters
    ----------
    sp : array_like
         wind speed [m s :sup:`-1`]
    z : float, array_like
        measurement height [m]
    Ta : array_like, optional for drag='smith'
         air temperature [:math:`^\\circ` C]
    Returns
    -------
    cd : float, array_like
         neutral drag coefficient at 10 m
    u10 : array_like
          wind speed at 10 m [m s :sup:`-1`]
    """
    # convert input to numpy array
    sp, z, Ta = np.asarray(sp), np.asarray(z), np.asarray(Ta)

    tol = 0.00001  # Iteration end point.

    a = np.log(z / 10.) / kappa  # Log-layer correction factor.
    u10o = np.zeros(sp.shape)
    cd = 1.15e-3 * np.ones(sp.shape)
    u10 = sp / (1 + a * np.sqrt(cd))
    ii = np.abs(u10 - u10o) > tol

    while np.any(ii):
        u10o = u10
        cd = (4.9e-4 + 6.5e-5 * u10o)  # Compute cd(u10).
        cd[u10o < 10.15385] = 1.15e-3
        u10 = sp / (1 + a * np.sqrt(cd))  # Next iteration.
        # Keep going until iteration converges.
        ii = np.abs(u10 - u10o) > tol

    return cd, u10

def stress(sp, z=10., rho_air=1.22, Ta=10.):
    """Computes the neutral wind stress.
    Parameters
    ----------
    sp : array_like
         wind speed [m s :sup:`-1`]
    z : float, array_like, optional
        measurement height [m]
    rho_air : array_like, optional
           air density [kg m :sup:`-3`]
    Ta : array_like, optional
         air temperature [:math:`^\\circ` C]
    Returns
    -------
    tau : array_like
          wind stress  [N m :sup:`-2`]
    """
    z, sp = np.asarray(z), np.asarray(sp)
    Ta, rho_air = np.asarray(Ta), np.asarray(rho_air)

    # Find cd and ustar.
    cd, sp = cdn(sp, z)
    tau = rho_air * (cd * sp ** 2)

    return tau
