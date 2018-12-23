from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import utils as tutils,covtools,ilc
from szar import foregrounds as fg

c = tutils.Config()

shape,wcs = enmap.read_map_geometry("datacov.hdf")
shape = shape[-2:]
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)
kbeams = []
freqs = []
for array in c.arrays:
    freqs.append(c.darrays[array]['freq'])
    kbeams.append(maps.gauss_beam(modlmap,c.darrays[array]['beam']))

chunk_size = 2000000

theory = cosmology.default_theory()

cov = enmap.read_map("datacov.hdf")
cov = maps.symmat_from_data(cov)

tcmb = 2.726e6
yresponses = fg.ffunc(freqs)*tcmb
cresponses = yresponses*0.+1.
responses={'tsz':yresponses,'cmb':cresponses}
ilcgen = ilc.chunked_ilc(modlmap,np.stack(kbeams),cov,chunk_size,responses=responses,invert=True)

snoise = enmap.empty((Ny*Nx),wcs)
cnoise = enmap.empty((Ny*Nx),wcs)
for chunknum,(hilc,selchunk) in enumerate(ilcgen):
    print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
    snoise[selchunk] = hilc.standard_noise("cmb")
    cnoise[selchunk] = hilc.constrained_noise("cmb","tsz")

del ilcgen,cov
snoise = snoise.reshape((Ny,Nx))
cnoise = cnoise.reshape((Ny,Nx))
bin_edges = np.arange(80,8000,80)
ells = np.arange(0,8000,1)
binner = stats.bin2D(modlmap,bin_edges)
cents,s1d = binner.bin(snoise)
cents,c1d = binner.bin(cnoise)

pl = io.Plotter(yscale='log',scalefn=lambda x:x**2./np.pi)
pl.add(cents,s1d)
pl.add(cents,c1d)
pl.add(ells,theory.lCl('TT',ells))
pl.done("snoise_data.png")
