from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,reproject,mpi,utils
import numpy as np
import os,sys
import healpy as hp
from soapack import interfaces as sints
from enlib import bench

"""
This script loads a Planck Galactic mask
at a specified fsky, combines it with a
coadd mask to make the most restrictive
wide patch mask.
"""

apod = 3.0
dfact = 16
ivar = enmap.downgrade(enmap.read_map("/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release2/act_planck_s08_s18_cmb_f090_daynight_ivar.fits",sel=np.s_[0,...]),dfact,op=np.sum)
rms = maps.rms_from_ivar(ivar)
ivar[rms>60] = 0
ivar[rms<0.01] = 0

opath = "/scratch/r/rbond/msyriac/data/act/dr5_masks/planck_gal/"
oshape,owcs = enmap.read_map_geometry(f'{opath}planck_gal_mask_equ_GAL070.fits')

ivar = maps.binary_mask(ivar,ivar[ivar>0].min())

#for gcut in ['GAL090','GAL097','GAL099']:
for gcut in ['GAL070','GAL060','GAL040','GAL020','GAL080']:
    mask = enmap.downgrade(enmap.read_map(f'{opath}planck_gal_mask_equ_{gcut}.fits'),dfact)
    io.hplot(mask*ivar,f'{opath}mivar_{gcut}',mask=0,grid=True,ticks=20)

    mask = (mask * ivar).astype(np.bool)
    r = apod * utils.degree
    with bench.show("dist"):
        mask = 0.5*(1-np.cos(mask.distance_transform(rmax=r)*(np.pi/r)))

    mask = enmap.upgrade(mask,dfact)
    print(mask.wcs)
    print(owcs)

    io.hplot(mask,f'{opath}../masks_20200723/wide_mask_{gcut}',downgrade=8,grid=True,ticks=20)
    # io.hplot(mask,f'{opath}../masks_20200723/wide_mask_{gcut}',mask=0,grid=True,ticks=20)
    enmap.write_map(f"{opath}../masks_20200723/wide_mask_{gcut}_apod_{apod:.2f}_deg.fits",mask)
