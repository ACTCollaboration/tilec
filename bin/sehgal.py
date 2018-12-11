from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,reproject,curvedsky
import numpy as np
import os,sys
import healpy as hp
from pixell.reproject import distribute, populate

croot = "/home/msyriac/data/sims/sehgal/colin/"

# cmap = enmap.read_map("cmap.fits")
# hfile = croot + "healpix_4096_KappaeffLSStoCMBfullsky.fits" #"143_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits"
# hmap = hp.read_map(hfile)

# pcls = hp.alm2cl(curvedsky.map2alm(cmap,lmax=6000))
# hcls = hp.anafast(hmap,lmax=6000)

# pells = range(len(pcls))
# hells = range(len(hcls))

# pl = io.Plotter()
# pl.add(pells,pcls*pells,label='car')
# pl.add(hells,hcls*hells,label='healpix')
# pl.done("cls.png")



# sys.exit()

        
res = 1.0
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.))

hfile = croot + "healpix_4096_KappaeffLSStoCMBfullsky.fits" #"143_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits"
# hfile = croot + "143_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits"

box = np.deg2rad([[-1,-1],[10,10]])
#shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.))
# shape,wcs = enmap.geometry(pos=box,res=np.deg2rad(res/60.))

hmap = hp.read_map(hfile)
ofunc = lambda shape,wcs : reproject.enmap_from_healpix_interp(hmap, shape, wcs, rot=None,interpolate=False)

cmap = populate(shape,wcs,ofunc,maxpixy=2000,maxpixx=2000) #enmap.downgrade(populate(shape,wcs,ofunc),2)
# enmap.write_map("cmap.fits",cmap)

# io.plot_img(cmap,"fmap.png")
cmap = cmap.submap(box)
io.hplot(cmap,"cmap",grid=False)

sys.exit()
pcls = hp.alm2cl(curvedsky.map2alm(cmap,lmax=6000))
hcls = hp.anafast(hmap,lmax=6000)

pells = range(len(pcls))
hells = range(len(hcls))

pl = io.Plotter()
pl.add(pells,(pcls-hcls)/pcls,label='car')
#pl.add(pells,pcls*pells,label='car')
#pl.add(hells,hcls*hells,label='healpix')
pl._ax.set_xlim(200,4000)
pl._ax.set_ylim(-0.2,0.1)
pl.done("cls.png")
