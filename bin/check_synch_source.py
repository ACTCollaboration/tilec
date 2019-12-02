from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,reproject
import numpy as np
import os,sys

fg = "tsz"

#ppath = "/scratch/r/rbond/msyriac/data/depot/tilec/map_rad_src_test_v1.0.0_rc_deep56/"
ppath = "/scratch/r/rbond/msyriac/data/depot/tilec/map_%s_src_test_v1.0.0_rc_deep56/" % fg
dec = 0
ra = 10

#imap = enmap.read_map(ppath + "tilec_single_tile_deep56_comptony_map_rad_src_test_v1.0.0_rc.fits")
imap = enmap.read_map(ppath + "tilec_single_tile_deep56_comptony_map_%s_src_test_v1.0.0_rc.fits" % fg)

cut = reproject.cutout(imap,ra=np.deg2rad(ra),dec=np.deg2rad(dec),npix=80) #* 1e6
_,nwcs = enmap.geometry(pos=(0,0),shape=cut.shape,res=np.deg2rad(0.5/60.))

print(cut.shape)

io.hplot(enmap.enmap(cut,nwcs),"%scut" % fg,color='gray',colorbar=True,ticks=5,tick_unit='arcmin',font_size=12,upgrade=4,quantile=1e-3)
