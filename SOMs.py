#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = "Self-Organising Map for North Atlantic"
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@student.unsw.edu.au"

#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from sompy.sompy import SOMFactory

#==============================================================================

def soms(data):
    """
    Input: 3-D array (nt,ny,nx)
    """
    #Reshape data
    nt,ny,nx = data.shape
    data = np.reshape(data, [nt, ny*nx], order='F')

    sm = SOMFactory().build(data, mapsize=(5,5), normalization=None, initialization='pca')
    sm.train(n_job=-1, verbose=False, train_rough_len=20, train_finetune_len=10)

    return sm

def k_means(data):
    """
    Input: 3-D array (nt,ny,nx)
    """
    nt,ny,nx = data.shape
    data = np.reshape(data, [nt, ny*nx], order='F')
    mk = KMeans(n_clusters=4, random_state=0, n_jobs=-1).fit(data)

    return mk

def get_cluster_fraction(m, label):
        return (m.labels_==label).sum()/(m.labels_.size*1.0)
#=================================================================
## Execute script
#
# if __name__ == "__main__":
#
#     codebook =  sm.codebook.matrix
#     print(codebook.shape)
#
#     x,y = np.meshgrid(da.X, da.Y)
#     proj = ccrs.Orthographic(0,45)
#     fig, axes = plt.subplots(5,5, figsize=(15,15), subplot_kw=dict(projection=proj))
#
#     for i in range(sm.codebook.nnodes):
#         onecen = codebook[i,:].reshape(ny,nx, order='F')
#         cs = axes.flat[i].contourf(x, y, onecen,
#                                    levels=np.arange(-150, 151, 30),
#                                    transform=ccrs.PlateCarree(),
#                                    cmap='RdBu_r')
#
#         cb=fig.colorbar(cs, ax=axes.flat[i], shrink=0.8, aspect=20)
#         cb.set_label('[unit: m]',labelpad=-7)
#         axes.flat[i].coastlines()
#         axes.flat[i].set_global()
#
#     #Check amounts of each regime using hitsmap
#     from sompy.visualization.bmuhits import BmuHitsView
#
#     vhts  = BmuHitsView(5, 5, "Amount of each regime",text_size=12)
#     vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="RdBu_r", logaritmic=False)
#
#     #PLot k-mean
#     x,y = np.meshgrid(da.X, da.Y)
#     proj = ccrs.Orthographic(0,45)
#     fig, axes = plt.subplots(2,2, figsize=(8,8), subplot_kw=dict(projection=proj))
#     regimes = ['NAO$^-$', 'NAO$^+$', 'Blocking', 'Atlantic Ridge']
#     tags = list('abcd')
#     for i in range(mk.n_clusters):
#         onecen = mk.cluster_centers_[i,:].reshape(ny,nx, order='F')
#         cs = axes.flat[i].contourf(x, y, onecen,
#                                    levels=np.arange(-150, 151, 30),
#                                    transform=ccrs.PlateCarree(),
#                                    cmap='RdBu_r')
#
#         cb=fig.colorbar(cs, ax=axes.flat[i], shrink=0.8, aspect=20)
#         cb.set_label('[unit: m]',labelpad=-7)
#         axes.flat[i].coastlines()
#         axes.flat[i].set_global()
#
#         title = '{}, {:4.1f}%'.format(regimes[i], get_cluster_fraction(mk, i)*100)
#         axes.flat[i].set_title(title)
#         plt.text(0, 1, tags[i], 
#                  transform=axes.flat[i].transAxes,
#                  va='bottom',
#                  fontsize=plt.rcParams['font.size']*2,
#              fontweight='bold')
