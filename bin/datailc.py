from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
from enlib import bench
import numpy as np
import os,sys
from tilec import utils as tutils,covtools,ilc,fg as tfg
from szar import foregrounds as fg

c = tutils.Config()

shape,wcs = enmap.read_map_geometry("output/datacov.hdf")
shape = shape[-2:]
Ny,Nx = shape
modlmap = enmap.modlmap(shape,wcs)
kbeams = []
freqs = []
kcoadds = []
for i,array in enumerate(c.arrays):
    freqs.append(c.darrays[array]['freq'])
    kbeams.append(c.get_beam(modlmap,array))
    kcoadds.append(c.load(i,skip_splits=True)[1].copy())
kcoadds = enmap.enmap(np.stack(kcoadds),c.wcs)
chunk_size = 1000000

theory = cosmology.default_theory()

cov = enmap.read_map("output/datacov.hdf")
cov = maps.symmat_from_data(cov)

tcmb = 2.726e6
yresponses = tfg.get_mix(freqs, "tSZ")
cresponses = tfg.get_mix(freqs, "CMB")
dresponses = tfg.get_mix(freqs, "CIB")

responses={'tsz':yresponses,'cmb':cresponses,'dust':dresponses}
ilcgen = ilc.chunked_ilc(modlmap,np.stack(kbeams),cov,chunk_size,responses=responses,invert=True)

snoise = enmap.empty((Ny*Nx),wcs)
cnoise = enmap.empty((Ny*Nx),wcs)
smap = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
cmap = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
ysmap = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
ycmap = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
ycmap_d = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
ycmap_dc = enmap.empty((Ny*Nx),wcs,dtype=np.complex128)
kcoadds = kcoadds.reshape((len(c.arrays),Ny*Nx))
for chunknum,(hilc,selchunk) in enumerate(ilcgen):
    print("ILC on chunk ", chunknum+1, " / ",int(modlmap.size/chunk_size)+1," ...")
    snoise[selchunk] = hilc.standard_noise("cmb")
    cnoise[selchunk] = hilc.constrained_noise("cmb","tsz")
    smap[selchunk] = hilc.standard_map(kcoadds[...,selchunk],"cmb")
    cmap[selchunk] = hilc.constrained_map(kcoadds[...,selchunk],"cmb","tsz")
    ysmap[selchunk] = hilc.standard_map(kcoadds[...,selchunk],"tsz")
    ycmap[selchunk] = hilc.constrained_map(kcoadds[...,selchunk],"tsz","cmb")
    ycmap_d[selchunk] = hilc.constrained_map(kcoadds[...,selchunk],"tsz","dust")
    # ycmap_dc[selchunk] = np.nan_to_num(hilc.multi_constrained_map(kcoadds[...,selchunk],"tsz",["dust","cmb"]))

del ilcgen,cov
snoise = enmap.enmap(snoise.reshape((Ny,Nx)),wcs)
cnoise = enmap.enmap(cnoise.reshape((Ny,Nx)),wcs)
ksmap = enmap.enmap(smap.reshape((Ny,Nx)),wcs)
kcmap = enmap.enmap(cmap.reshape((Ny,Nx)),wcs)
yksmap = enmap.enmap(ysmap.reshape((Ny,Nx)),wcs)
ykcmap = enmap.enmap(ycmap.reshape((Ny,Nx)),wcs)
ykcmap_d = enmap.enmap(ycmap_d.reshape((Ny,Nx)),wcs)
# ykcmap_dc = enmap.enmap(ycmap_dc.reshape((Ny,Nx)),wcs)
enmap.write_map("snoise.fits",snoise)
enmap.write_map("cnoise.fits",cnoise)
enmap.write_map("ksmap.fits",enmap.enmap(c.fc.ifft(ksmap).real,wcs))
enmap.write_map("kcmap.fits",enmap.enmap(c.fc.ifft(kcmap).real,wcs))
io.plot_img(maps.ftrans(snoise),"snoise2d.png",aspect='auto')
io.plot_img(maps.ftrans(cnoise),"cnoise2d.png",aspect='auto')
bin_edges = np.arange(80,8000,80)
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
pl.done("snoise_data.png")

smap = enmap.enmap(c.fc.ifft(ksmap).real,wcs)
cmap = enmap.enmap(c.fc.ifft(kcmap).real,wcs)
ysmap = enmap.enmap(c.fc.ifft(yksmap).real,wcs)
ycmap = enmap.enmap(c.fc.ifft(ykcmap).real,wcs)
ycmap_d = enmap.enmap(c.fc.ifft(ykcmap_d).real,wcs)
# ycmap_dc = enmap.enmap(c.fc.ifft(ykcmap_dc).real,wcs)
io.hplot(smap,"usmap")
io.hplot(cmap,"ucmap")
io.plot_img(smap,"umsmap.png",lim=300)
io.plot_img(cmap,"umcmap.png",lim=300)

#sbeam = 1.4
#cbeam = 2.2
skbeam = c.get_beam(modlmap,'s15_pa3_150')
ckbeam = c.get_beam(modlmap,'s15_pa3_90')

# Uncomment this back
# smap = maps.filter_map(enmap.enmap(smap,wcs),skbeam)
# cmap = maps.filter_map(enmap.enmap(cmap,wcs),ckbeam)
# ysmap = maps.filter_map(enmap.enmap(ysmap,wcs),skbeam)
# ycmap = maps.filter_map(enmap.enmap(ycmap,wcs),ckbeam)
# ycmap_d = maps.filter_map(enmap.enmap(ycmap_d,wcs),ckbeam)


# ycmap_dc = maps.filter_map(enmap.enmap(ycmap_dc,wcs),ckbeam)
io.hplot(ysmap,"ysmap",grid=True)
io.hplot(ycmap,"ycmap",grid=True)
io.hplot(ycmap_d,"ycmap_d",grid=True)
io.hplot(smap,"smap",grid=True)
io.hplot(cmap,"cmap",grid=True)
io.plot_img(smap,"msmap.png",lim=300)
io.plot_img(cmap,"mcmap.png",lim=300)
enmap.write_map("constrained_y_map.fits",ycmap)
enmap.write_map("constrained_y_map_dedust.fits",ycmap_d)
# enmap.write_map("constrained_y_map_deboth.fits",ycmap_dc)
enmap.write_map("standard_y_map.fits",ysmap)
enmap.write_map("constrained_cmb_map.fits",cmap)
enmap.write_map("standard_cmb_map.fits",smap)
# io.hplot(ycmap_dc,"ycmap_dc",grid=True)

bin_edges = np.arange(80,15000,80)
binner = stats.bin2D(modlmap,bin_edges)
#kbeam = maps.gauss_beam(modlmap,sbeam)
cents,as1d = binner.bin(snoise*skbeam**2.)
#kbeam = maps.gauss_beam(modlmap,cbeam)
cents,ac1d = binner.bin(cnoise*ckbeam**2.)

pl = io.Plotter(yscale='log')
pl.add(cents,as1d,ls="-")
pl.add(cents,ac1d,ls="-")
pl.done("snoise_data_beamed.png")
