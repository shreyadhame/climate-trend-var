import numpy as np

#-------------------------------------------------------------------------------------------------------------------
# WAVE_SIGNIF  Significance testing for the 1D Wavelet transform WAVELET
#
#   SIGNIF = wave_signif(Y,DT,SCALE,SIGTEST,LAG1,SIGLVL,DOF,MOTHER,PARAM)
#
# INPUTS:
#
#    Y = the time series, or, the VARIANCE of the time series.
#        (If this is a single number, it is assumed to be the variance...)
#    DT = amount of time between each Y value, i.e. the sampling time.
#    SCALE = the vector of scale indices, from previous call to WAVELET.
#
#
# OUTPUTS:
#
#    SIGNIF = significance levels as a function of SCALE
#    FFT_THEOR = output theoretical red-noise spectrum as fn of PERIOD
#
#
# OPTIONAL INPUTS:
#    SIGTEST = 0, 1, or 2.    If omitted, then assume 0.
#
#         If 0 (the default), then just do a regular chi-square test,
#             i.e. Eqn (18) from Torrence & Compo.
#         If 1, then do a "time-average" test, i.e. Eqn (23).
#             In this case, DOF should be set to NA, the number
#             of local wavelet spectra that were averaged together.
#             For the Global Wavelet Spectrum, this would be NA=N,
#             where N is the number of points in your time series.
#         If 2, then do a "scale-average" test, i.e. Eqns (25)-(28).
#             In this case, DOF should be set to a
#             two-element vector [S1,S2], which gives the scale
#             range that was averaged together.
#             e.g. if one scale-averaged scales between 2 and 8,
#             then DOF=[2,8].
#
#    LAG1 = LAG 1 Autocorrelation, used for SIGNIF levels. Default is 0.0
#
#    SIGLVL = significance level to use. Default is 0.95
#
#    DOF = degrees-of-freedom for signif test.
#         IF SIGTEST=0, then (automatically) DOF = 2 (or 1 for MOTHER='DOG')
#         IF SIGTEST=1, then DOF = NA, the number of times averaged together.
#         IF SIGTEST=2, then DOF = [S1,S2], the range of scales averaged.
#
#       Note: IF SIGTEST=1, then DOF can be a vector (same length as SCALEs),
#            in which case NA is assumed to vary with SCALE.
#            This allows one to average different numbers of times
#            together at different scales, or to take into account
#            things like the Cone of Influence.
#            See discussion following Eqn (23) in Torrence & Compo.
#
#    GWS = global wavelet spectrum, a vector of the same length as scale.
#          If input then this is used as the theoretical background spectrum,
#          rather than white or red noise.


def wave_signif(Y, dt, scale, sigtest=0, lag1=0.0, siglvl=0.95,
    dof=None, mother='MORLET', param=None, gws=None):
    n1 = len(np.atleast_1d(Y))
    J1 = len(scale) - 1
    s0 = np.min(scale)
    dj = np.log2(scale[1] / scale[0])

    if n1 == 1:
        variance = Y
    else:
        variance = np.std(Y) ** 2

    # get the appropriate parameters [see Table(2)]
    if mother == 'MORLET':  # ----------------------------------  Morlet
        empir = ([2., -1, -1, -1])
        if param is None:
            param = 6.
            empir[1:] = ([0.776, 2.32, 0.60])
        k0 = param
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))  # Scale-->Fourier [Sec.3h]
    elif mother == 'PAUL':
        empir = ([2, -1, -1, -1])
        if param is None:
            param = 4
            empir[1:] = ([1.132, 1.17, 1.5])
        m = param
        fourier_factor = (4 * np.pi) / (2 * m + 1)
    elif mother == 'DOG':  # -------------------------------------Paul
        empir = ([1., -1, -1, -1])
        if param is None:
            param = 2.
            empir[1:] = ([3.541, 1.43, 1.4])
        elif param == 6:  # --------------------------------------DOG
            empir[1:] = ([1.966, 1.37, 0.97])
        m = param
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
    else:
        print('Mother must be one of MORLET, PAUL, DOG')

    period = scale * fourier_factor
    dofmin = empir[0]  # Degrees of freedom with no smoothing
    Cdelta = empir[1]  # reconstruction factor
    gamma_fac = empir[2]  # time-decorrelation factor
    dj0 = empir[3]  # scale-decorrelation factor

    freq = dt / period  # normalized frequency

    if gws is not None:   # use global-wavelet as background spectrum
        fft_theor = gws
    else:
        fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 *
            np.cos(freq * 2 * np.pi) + lag1 ** 2)  # [Eqn(16)]
        fft_theor = variance * fft_theor  # include time-series variance

    signif = fft_theor
    if dof is None:
        dof = dofmin

    if sigtest == 0:  # no smoothing, DOF=dofmin [Sec.4]
        dof = dofmin
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = fft_theor * chisquare  # [Eqn(18)]
    elif sigtest == 1:  # time-averaged significance
        if len(np.atleast_1d(dof)) == 1:
            dof = np.zeros(J1) + dof
        dof[dof < 1] = 1
        dof = dofmin * np.sqrt(1 + (dof * dt / gamma_fac / scale) ** 2)  # [Eqn(23)]
        dof[dof < dofmin] = dofmin   # minimum DOF is dofmin
        for a1 in range(0, J1 + 1):
            chisquare = chisquare_inv(siglvl, dof[a1]) / dof[a1]
            signif[a1] = fft_theor[a1] * chisquare
    elif sigtest == 2:  # time-averaged significance
        if len(dof) != 2:
            print('ERROR: DOF must be set to [S1,S2], the range of scale-averages')
        if Cdelta == -1:
            print('ERROR: Cdelta & dj0 not defined for ' +
                    mother + ' with param = ' + str(param))

        s1 = dof[0]
        s2 = dof[1]
        avg = np.logical_and(scale >= 2, scale < 8)  # scales between S1 & S2
        navg = np.sum(np.array(np.logical_and(scale >= 2, scale < 8), dtype=int))
        if navg == 0:
            print('ERROR: No valid scales between ' + str(s1) + ' and ' + str(s2))
        Savg = 1. / np.sum(1. / scale[avg])  # [Eqn(25)]
        Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)  # power-of-two midpoint
        dof = (dofmin * navg * Savg / Smid) * \
                np.sqrt(1 + (navg * dj / dj0) ** 2)  # [Eqn(28)]
        fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])  # [Eqn(27)]
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare  # [Eqn(26)]
    else:
        print('ERROR: sigtest must be either 0, 1, or 2')

    return signif,fft_theor,variance

#-------------------------------------------------------------------------------------------------------------------
# CHISQUARE_INV  Inverse of chi-square cumulative distribution function (cdf).
#
#   X = chisquare_inv(P,V) returns the inverse of chi-square cdf with V
#   degrees of freedom at fraction P.
#   This means that P*100 percent of the distribution lies between 0 and X.
#
#   To check, the answer should satisfy:   P==gammainc(X/2,V/2)

# Uses FMIN and CHISQUARE_SOLVE


def chisquare_inv(P, V):

    if (1 - P) < 1E-4:
        print('P must be < 0.9999')

    if P == 0.95 and V == 2:  # this is a no-brainer
        X = 5.9915
        return X

    MINN = 0.01  # hopefully this is small enough
    MAXX = 1  # actually starts at 10 (see while loop below)
    X = 1
    TOLERANCE = 1E-4  # this should be accurate enough

    while (X + TOLERANCE) >= MAXX:  # should only need to loop thru once
        MAXX = MAXX * 10.
    # this calculates value for X, NORMALIZED by V
        X = fminbound(chisquare_solve, MINN, MAXX, args=(P, V), xtol=TOLERANCE)
        MINN = MAXX

    X = X * V  # put back in the goofy V factor

    return X  # end of code

#-------------------------------------------------------------------------------------------------------------------
# CHISQUARE_SOLVE  Internal function used by CHISQUARE_INV
    #
    #   PDIFF=chisquare_solve(XGUESS,P,V)  Given XGUESS, a percentile P,
    #   and degrees-of-freedom V, return the difference between
    #   calculated percentile and P.

    # Uses GAMMAINC
    #
    # Written January 1998 by C. Torrence

    # extra factor of V is necessary because X is Normalized


def chisquare_solve(XGUESS, P, V):

    PGUESS = gammainc(V/2, V*XGUESS/2)  # incomplete Gamma function

    PDIFF = np.abs(PGUESS - P)            # error in calculated P

    TOL = 1E-4
    if PGUESS >= 1-TOL:  # if P is very close to 1 (i.e. a bad guess)
        PDIFF = XGUESS   # then just assign some big number like XGUESS

    return PDIFF
