from __future__ import print_function
from pixell import enmap,curvedsky as cs,reproject
import numpy as np
import os,sys
from soapack import interfaces as sints
import healpy as hp

dm = sints.DR5()
shape,wcs = sints.get_advact_geometry()
alm = hp.read_alm("/scratch/r/rbond/msyriac/data/planck/data/pr2/COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00_alm.fits")
omap = reproject.enmap_from_healpix(alm, shape, wcs, ncomp=1, unit=1, lmax=0,
                                    rot="gal,equ", first=0, is_alm=True, return_alm=False, f_ell=None)
enmap.write_map('/scratch/r/rbond/msyriac/data/planck/data/pr2/COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00_advact.fits',omap)
