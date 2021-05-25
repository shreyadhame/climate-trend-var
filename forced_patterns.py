import sys
sys.path.append('/srv/ccrc/data25/z5166746/Scripts/my_scripts/')
import bottleneck as bn
import gc
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, filtfilt

from lanczosfilter import *
from lowfreq_components import *

#=================================================================
def forced_pattern_analysis(X, Xe, truncation, scale, Covtot=None):
    """
    Truncated forced pattern analysis

    INPUT:-
    X : 2D array with time variations along the first dimension and spatial
    variations along the second dimension. Ensemble members
    are concatenated in time.

    Xe : 2D ensemble-mean data matrix with time variantions along the
    first dimension and spatial variations along the second dimension

    truncation : number of principal components / EOFs to include in
    the analysis, or (if less than 1) the fraction of variance to retain

    scale :  (optional) a scale vector, which for geospatial data should be
    equal to the square root of grid cell area. The default value is one
    for all grid points.

    Covtot : (optional) the covariance matrix associated with the data in
    X. If not specified, COVTOT will be computed from X.

    OUTPUT:-
    fingerprints : matrix containing the canonical weight vectors as
    columns. FPs is a matrix containing the dual vectors of the
    canonical weight vectors as rows. These are the so-called forced
    patterns (FPs). S is a vector measuring the ratio of ensemble-mean
    signal to total variance for each forced pattern

    pvar : percentage of total sample variation accounted for
    by each of the EOFs. PCS is a matrix containing the principal
    component time series as columns. EOF is a matrix containing the
    EOFs, the principal component patterns, as rows. The scalar N
    is the rank at which the PCA was truncated.

    S : vector containing the ratio of ensemble-mean signal to total
    variance for each forced pattern

    pvar_fps : vector of the variance associated with
    each forced pattern as a fraction of the total variance. Note that
    the FPs are not orthogonal, so these values need not add to the
    total variance in the first N principal components.

    S_EOFS and PVAR are equivalent to S and PVAR_FPS respectively, but
    for the original EOFs.
    """

    #Check shape of input argument
    if len(X.shape) != 2:
        raise ValueError('Data array must be 2D')
    else:
        pass

    n,p = Xe.shape #(792, 108000)

    ne = int(X.shape[0]/n)

    if np.isnan(X).any(): #there are missing values in X
        Xm = np.nanmean(X)
        Xem = np.nanmean(Xe)
    else:
        Xm = X.mean()
        Xem = Xe.mean()

    X = X - np.tile(Xm, (int(n*ne),1))
    Xe = Xe - np.tile(Xem, (int(n),1))

    Covtot = np.cov(X.astype('float16'),rowvar=False)

    if Covtot.shape != (p,p):
        raise ValueError('Covariance matrix must have same dimension as data')
    else:
        pass

    scale=scale.T
    if np.max(np.shape(scale)) != p:
        raise ValueError('Scale vector must have same dimension as data')
    Xs = X * np.tile(scale,(n*ne,1))
    Xes = Xe * np.tile(scale,(n,1))

    del X, Xm, Xe, Xem
    gc.collect()

    #eigendecomposition of covariance matrix
    Covtot = np.tile(scale.T,(1,p))*Covtot*np.tile(scale,(p,1))
    print(Covtot.shape)
    # Covtot = Covtot.map_blocks(sparse.COO)
    pcvec,evl,rest = peigs(Covtot, np.min([n-1, p]))
    trCovtot = np.trace(Covtot)
    print(trCovtot.shape)
    #percent of total sample variation accounted for by each EOF
    pvar = evl/trCovtot * 100
    #principal component time series
    pcs = np.dot(Xs,pcvec)
    pces = np.dot(Xes,pcvec)
    s_eofs = np.var(pces,axis=(0,1))/np.var(pcs,axis=(0,1))
    #return EOFs in original scaling as patterns (row vectors)
    EOF = pcvec.T/np.tile(scale,(rest,1))
    print(EOF.shape)
    #truncation of EOFs
    if truncation < 1:
      #using basic % variance criterion, where truncation gives the
      #fraction of variance to be included in the EOF truncation
      truncation = truncation*100
      cum_pvar = np.cumsum(pvar)
      N = np.where(abs(cum_pvar-truncation) == np.min(abs(cum_pvar-truncation)))

    else:
        if (truncation - np.round(truncation)) != 0:
            raise ValueError('Truncation must be fraction of total variance included\
            in EOF truncation or integer number of EOFs')
        #using specified truncation level
        N = truncation

    #this section can be modified to use a specific EOF truncation
    #criterion, right now the truncation number is specified as input
    #TRUNCATION

    #Whitening transformation
    #multiplication factor for principal components in whitening
    #transformation (such that they have unit variance)
    f = np.sqrt(evl[:N])

    #get transformation matrices that transform original variables to whitened
    #variables and back
    S = np.dot(pcvec[:,:N] , np.diag(1./f))
    Sadj = np.dot(np.diag(f) , pcvec[:,:N].T)

    #whiten variables such that cov(Y) = I*n(n-1)
    Y = np.dot(Xes, S)

    #slow covarance matrix of whitened variables
    #(i.e. covariance matrix of filtered and whitened principal components)
    Gamma = np.cov(Y,rowvar=False)

    #SVD of slow covariance matrix (such that r are eigenvalues and V are eigenvectors)
    U, s, V = csvd(Gamma)

    # fingerprint patterns (canonical vectors) and forced patterns (FPs) in original scaling
    fingerprints = np.tile(scale.T,(1,N))*np.dot(S,V)
    FPs = np.dot(V.T, Sadj)/np.tile(scale,(N,1))
    print(FPs.shape)
    #choose signs of patterns, weights, eofs, and pcs such that the
    #scalar product of the vectors and the scale vector is positive
    for j in range(FPs.shape[0]):
        if np.dot(FPs[j,:][np.newaxis,...],scale.T) < 0.:
            FPs[j,:] = -FPs[j,:]
            fingerprints[:,j] = -fingerprints[:,j]

    for j in range(EOF.shape[0]):
        if np.dot(EOF[j,:][np.newaxis,...],scale.T) < 0:
            EOF[j,:] = -EOF[j,:]
            pcs[:,j] = -pcs[:,j]

    #timeseries
    Xs = Xs/np.tile(scale,(n*ne,1))

    tk = np.dot(Xs,fingerprints)

    #fraction of variance in forced patterns
    w = fingerprints/np.tile(scale.T,(1,N))
    p = FPs*np.tile(scale,(N,1))

    tot_var = np.diag(np.dot(np.dot(p,Covtot),w))/np.diag(np.dot(p,w))

    pvar_FPs = tot_var/trCovtot*100

    return tk, FPs, fingerprints, s, pvar, pcs, EOF, N, pvar_FPs, s_eofs
