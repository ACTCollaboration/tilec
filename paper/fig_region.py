from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys

do_reconvolve = True

#region = "boss"
region = "deep56"
version = "map_v1.1.1_joint"
cversion = "v1.1.1"
yname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_comptony_%s.fits" % (version,region,region,version)
ybname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_comptony_%s_beam.txt" % (version,region,region,version)

cname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_deprojects_comptony_%s.fits" % (version,region,region,version)
cbname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_deprojects_comptony_%s_beam.txt" % (version,region,region,version)

sname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_%s.fits" % (version,region,region,version)
sbname = "/scratch/r/rbond/msyriac/data/depot/tilec/v1.0.0_rc_20190919/../%s_%s/tilec_single_tile_%s_cmb_%s_beam.txt" % (version,region,region,version)

ymap = enmap.read_map(yname)
modlmap = ymap.modlmap()
cmap = enmap.read_map(cname)
smap = enmap.read_map(sname)


def reconvolve(x,ybname,fwhm):
    if not(do_reconvolve): return x
    ls,bells = np.loadtxt(ybname,unpack=True)
    beam_rat = maps.gauss_beam(modlmap,fwhm)/maps.interp(ls,bells)(modlmap)
    beam_rat[~np.isfinite(beam_rat)] = 0
    return maps.filter_map(x,beam_rat)

ymap = reconvolve(ymap,ybname,fwhm=2.2) 
cmap = reconvolve(cmap,cbname,fwhm=2.2) 
smap = reconvolve(smap,sbname,fwhm=2.2) 

box = np.deg2rad(np.array([ [-2,12],
                            [2,18]] ))


enmap.write_map("ymap_%s.fits" % region,ymap)
enmap.write_map("smap_%s.fits" % region,smap)

enmap.write_map("ycutout.fits",ymap.submap(box))
io.hplot(ymap.submap(box),'fig_region_ymap',color='gray',grid=True,colorbar=True,annotate='paper/public_clusters.csv',min=-1.25e-5,max=3.0e-5)
io.hplot(cmap.submap(box),'fig_region_cmap',color='planck',grid=True,colorbar=True,range=300,annotate='paper/public_clusters.csv')
io.hplot(smap.submap(box),'fig_region_smap',color='planck',grid=True,colorbar=True,range=300,annotate='paper/public_clusters.csv')


"""
Add 
c -1.2500000 14.0700000 0 0 35 3 red
for Abell 119
"""
