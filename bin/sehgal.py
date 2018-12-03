from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,reproject
import numpy as np
import os,sys
import healpy as hp

croot = "/home/msyriac/data/sims/sehgal/colin/"


hfile = croot + "healpix_4096_KappaeffLSStoCMBfullsky.fits" #"143_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits"

res = 0.5
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.))

hmap = hp.read_map(hfile)
omap = reproject.enmap_from_healpix(hmap, shape, wcs, rot=None)


box = np.deg2rad([[-1,-1],[10,10]])
cmap = omap.submap(box)
io.hplot(cmap,"cmap")
