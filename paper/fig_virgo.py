from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from pixell import enmap,reproject
import numpy as np
import os,sys,shutil
from soapack import interfaces as sints
import healpy as hp

box = np.deg2rad([ [11,-173.5],
                   [13.5,-171] ])



rosat = enmap.read_map("/scratch/r/rbond/msyriac/data/for_sigurd/rosat_r7_boss.fits").submap(box)

print(rosat.shape,maps.resolution(rosat.shape,rosat.wcs) * 180.*60./np.pi)

version = "map_v1.0.0_rc_joint"
cversion = "v1.0.0_rc"
region = 'boss'
yname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/%s_%s/tilec_single_tile_%s_comptony_%s.fits" % (version,region,region,version)

ymap = enmap.read_map(yname).submap(box)
print(ymap.center()*180./np.pi)
rosat = enmap.project(rosat,ymap.shape,ymap.wcs)


io.hplot(np.log10(rosat),'fig_virgo_rosat')
#io.hplot(enmap.smooth_gauss(ymap,np.deg2rad(3./60.)),'fig_virgo_act',color='gray') #,min=-1.25e-5,max=3.0e-5
io.hplot(ymap,'fig_virgo_act',color='gray') #,min=-1.25e-5,max=3.0e-5

