from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,reproject
import numpy as np
import os,sys
import healpy as hp
from orphics.maps import distribute, populate

croot = "/home/msyriac/data/sims/sehgal/colin/"

        
res = 0.5
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.))

hfile = croot + "healpix_4096_KappaeffLSStoCMBfullsky.fits" #"143_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits"

box = np.deg2rad([[-1,-1],[10,10]])
#shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.))
# shape,wcs = enmap.geometry(pos=box,res=np.deg2rad(res/60.))

hmap = hp.read_map(hfile)
ofunc = lambda shape,wcs : reproject.enmap_from_healpix_interp(hmap, shape, wcs, rot=None,interpolate=True)

cmap = populate(shape,wcs,ofunc)

io.plot_img(cmap,"fmap.png")
cmap = cmap.submap(box)
io.hplot(cmap,"cmap")
