from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys

mask = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_mask.fits")
modlmap = mask.modlmap()

smap = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_map_v1.0.0_rc_joint.fits")
l,b = np.loadtxt("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_map_v1.0.0_rc_joint_beam.txt",unpack=True)
sbeam = maps.interp(l,b)(modlmap)
cmap = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_deprojects_comptony_map_v1.0.0_rc_joint.fits")
l,b = np.loadtxt("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_deprojects_comptony_map_v1.0.0_rc_joint_beam.txt",unpack=True)
cbeam = maps.interp(l,b)(modlmap)
nsmap = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_map_v1.0.0_rc_joint_noise.fits")
ncmap = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_deprojects_comptony_map_v1.0.0_rc_joint_noise.fits")
crossnoise = enmap.read_map("/scratch/r/rbond/msyriac/data/depot/tilec/map_v1.0.0_rc_joint_deep56/tilec_single_tile_deep56_cmb_deprojects_comptony_map_v1.0.0_rc_joint_cross_noise.fits")

w2 = np.mean(mask**2.)
ksmap = enmap.fft(smap,normalize='phys')
kcmap = enmap.fft(cmap,normalize='phys')
p2d = np.real(ksmap*kcmap.conj())/w2/sbeam/cbeam
sp2d = np.real(ksmap*ksmap.conj())/w2/sbeam**2.
cp2d = np.real(kcmap*kcmap.conj())/w2/cbeam**2.

bin_edges = np.arange(20,8000,20)
binner = stats.bin2D(modlmap,bin_edges)

cents,s1d = binner.bin(sp2d)
cents,c1d = binner.bin(cp2d)

cents,cross1d = binner.bin(p2d)
cents,tcross1d = binner.bin(crossnoise)
cents,ts1d = binner.bin(nsmap)
cents,tc1d = binner.bin(ncmap)

pl = io.Plotter(xyscale='linlog',scalefn = lambda x:x**2./2./np.pi,xlabel='l',ylabel='D')
#pl.add(cents,cross1d,color="C0")
#pl.add(cents,tcross1d,ls="--",color="C0")
#pl.add(cents,s1d,color="C1")
#pl.add(cents,ts1d,ls="--",color="C1")
pl.add(cents,c1d,color="C2")
pl.add(cents,tc1d,ls="--",color="C2")
pl.done("pscheck.png")
