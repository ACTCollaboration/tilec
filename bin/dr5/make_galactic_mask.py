from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,reproject,mpi
import numpy as np
import os,sys
import healpy as hp
from soapack import interfaces as sints

"""
This script loads a Planck Galactic mask
at a specified fsky, and saves it in the
AdvACT geometry.
"""

opath = "/scratch/r/rbond/msyriac/data/act/dr5_masks/planck_gal/"
oshape,owcs = sints.get_advact_geometry()
scale = 0.1
shape,wcs = enmap.scale_geometry(oshape, owcs, scale)

field = {'GAL020':0,'GAL040':1,'GAL060':2,'GAL070':3,'GAL080':4,'GAL090':5,'GAL097':6,'GAL099':7}
#field = {'GAL040':1}

comm = mpi.COMM_WORLD
nside = 2048
dtype = np.float32
rot = reproject.HRot(nside,shape=shape,wcs=wcs,rot='gal,equ',dtype=dtype)

if comm.rank==0:
    for gcut in sorted(field.keys()):
        print(gcut)
        hmap = hp.read_map("/scratch/r/rbond/msyriac/data/planck/data/pr3/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",field=field[gcut])
        umask = rot.ivar_hp_to_cyl(hmap,do_mask=True,extensive=True)[0]
        io.hplot(umask,f'{opath}planck_gal_mask_equ_{gcut}_high')
        umask = enmap.project(umask,oshape,owcs,order=1)
        mask = maps.binary_mask(umask)
        enmap.write_map(f'{opath}planck_gal_mask_equ_{gcut}.fits',mask)
        io.plot_img(mask,f'{opath}planck_gal_mask_equ_{gcut}.png')
