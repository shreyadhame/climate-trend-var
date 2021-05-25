#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Multiprocessing in Python"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
#General modules
from functools import partial
import multiprocessing
import numpy as np
import numpy.ma as ma

#============================================================
def chunks(matrix):
    """
    Converts 3D xarray into list of 1D arrays
    """
    #matrix_re = matrix.stack(z=(d1,d2))
    mat = np.array(matrix)
    matrix_re = np.reshape(mat,(mat.shape[0],mat.shape[1]*mat.shape[2]))
    matrix_chunks = matrix_re.T.tolist()
    # matrix_chunks=np.stack(matrix_re.T, axis=0)
    return matrix_chunks

def f_mp(f, iterable, ncores, a=0.10):
    """
    Multiprocessing function
    """
    pool = multiprocessing.Pool(processes=ncores)
    #result = pool.map(f,iterable)
    # else:
    func = partial(f, a=a)
    result = pool.map(func,iterable)
    pool.close()
    pool.join()
    return result

def calc_func(matrix,f,ncores,noutput,a=0.10):
    matrix_chunks = chunks(matrix)
    func_ = f_mp(f,matrix_chunks,ncores,a=a)
    #reshape
    C=[]
    for i in range(noutput):
        C.append(np.reshape([item[i] for item in func_],(matrix.shape[1],matrix.shape[2])))
    return C
