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
kcoadds = []
for i,array in enumerate(c.arrays):
    freqs.append(c.darrays[array]['freq'])
    kbeams.append(maps.gauss_beam(modlmap,c.darrays[array]['beam']))
    kcoadds.append(c.load(i,skip_splits=True)[1].copy())
kcoadds = enmap.enmap(np.stack(kcoadds),c.wcs)
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
smap = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
cmap = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
kcoadds = kcoadds.reshape((len(c.arrays),Ny*Nx))
for chunknum,(hilc,selchunk) in enumerate(ilcgen):
    print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
    snoise[selchunk] = hilc.standard_noise("cmb")
    cnoise[selchunk] = hilc.constrained_noise("cmb","tsz")
    smap[selchunk] = hilc.standard_map(kcoadds[...,selchunk],"cmb")
    cmap[selchunk] = hilc.constrained_map(kcoadds[...,selchunk],"cmb","tsz")

del ilcgen,cov
snoise = snoise.reshape((Ny,Nx))
cnoise = cnoise.reshape((Ny,Nx))
ksmap = enmap.enmap(smap.reshape((Ny,Nx)),wcs)
kcmap = enmap.enmap(cmap.reshape((Ny,Nx)),wcs)
bin_edges = np.arange(80,8000,80)
ells = np.arange(0,8000,1)
binner = stats.bin2D(modlmap,bin_edges)
cents,s1d = binner.bin(snoise)
cents,c1d = binner.bin(cnoise)

cents,as1d = binner.bin(c.fc.f2power(ksmap,ksmap))
cents,ac1d = binner.bin(c.fc.f2power(kcmap,kcmap))

pl = io.Plotter(yscale='log',scalefn=lambda x:x**2./np.pi)
pl.add(cents,as1d,ls="-")
pl.add(cents,ac1d,ls="-")
pl.add(cents,s1d,ls="--")
pl.add(cents,c1d,ls="--")
pl.add(ells,theory.lCl('TT',ells))
pl.done("snoise_data.png")

smap = enmap.enmap(c.fc.ifft(ksmap).real,wcs)
cmap = enmap.enmap(c.fc.ifft(kcmap).real,wcs)
io.hplot(smap,"usmap")
io.hplot(cmap,"ucmap")
io.plot_img(smap,"umsmap.png",lim=300)
io.plot_img(cmap,"umcmap.png",lim=300)

kbeam = maps.gauss_beam(modlmap,1.5)
smap = maps.filter_map(enmap.enmap(smap,wcs),kbeam)
kbeam = maps.gauss_beam(modlmap,3.0)
cmap = maps.filter_map(enmap.enmap(cmap,wcs),kbeam)
io.hplot(smap,"smap")
io.hplot(cmap,"cmap")
io.plot_img(smap,"msmap.png",lim=300)
io.plot_img(cmap,"mcmap.png",lim=300)
