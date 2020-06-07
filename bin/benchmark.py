from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
from soapack import interfaces as sints
from enlib import bench
import h5py
import healpy as hp

shape,wcs = sints.get_advact_geometry()
dtype = np.float32
imap = enmap.enmap(np.random.random(shape).astype(dtype),wcs)
print(imap.dtype)
lmax = 8192 * 3

# with bench.show("sht"):
#     alm = cs.map2alm(imap,lmax=lmax)


root = "/scratch/r/rbond/msyriac/data/depot/tilec/benchmark/"


# with bench.show('write alm npy'):
#     np.save(f'{root}alm.npy',alm)

# with bench.show('write alm h5py'):
#     with h5py.File(f'{root}alm.h5', 'w') as hf:
#         hf.create_dataset("alm",  data=alm)

# with bench.show('write alm fits'):
#     hp.write_alm(f'{root}alm_hp.fits',alm,overwrite=True)






# with bench.show("isht"):
#     omap = cs.alm2map(alm,enmap.empty(shape,wcs,dtype=imap.dtype))

# print(alm.dtype,omap.dtype)
# print(alm.nbytes / (1024)**3, omap.nbytes / (1024)**3)


# with bench.show('write map fits'):
#     enmap.write_map(f'{root}map_enmap.fits',omap)

# with bench.show('write map npy'):
#     np.save(f'{root}map.npy',omap)

# with bench.show('write map h5py'):
#     with h5py.File(f'{root}map.h5', 'w') as hf:
#         hf.create_dataset("map",  data=omap)



with bench.show('read alm npy'):
    ralm = np.load(f'{root}alm.npy')


with bench.show('read alm h5py'):
    with h5py.File(f'{root}alm.h5', 'r') as hf:
        data = hf['alm'][:]

with bench.show('read alm fits'):
    ralm2 = hp.read_alm(f'{root}alm_hp.fits')


with bench.show('read map fits'):
    rmap = enmap.read_map(f'{root}map_enmap.fits')

with bench.show('read map npy'):
    rmap2 = np.load(f'{root}map.npy')

with bench.show('read map h5py'):
    with h5py.File(f'{root}map.h5', 'r') as hf:
        data2 = hf['map'][:]
