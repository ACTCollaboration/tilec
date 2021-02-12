from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,wcsutils
import numpy as np
import os,sys
import healpy as hp
import utils as cutils
from tilec import needlets as nd,pipeline,fg as tfg
from soapack import interfaces as sints
import h5py

version = 'test'
lmax_file = 'data/needlet_lmaxs_szmode.txt'
bound_file = 'data/needlet_bounds_szmode.txt'
fwhm = 1.7
qids = cutils.qids
opath = f'{cutils.opath}/{version}/'
dpath = opath

dm = sints.DR5()

cmap = enmap.read_map(f'{dpath}cmap.fits')
ymap = enmap.read_map(f'{dpath}ymap.fits')

print(ymap)
print(ymap.shape,ymap.wcs,ymap.dtype)
print(ymap.min(),ymap.max())


# io.plot_img(ymap,f'{dpath}ymap_lowres')
io.hplot(ymap,f'{dpath}ymap',grid=True,ticks=10,color='gray',downgrade=2,colorbar=True)
io.hplot(cmap,f'{dpath}cmap',grid=True,ticks=10,downgrade=2,colorbar=True)
