from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import soapack.interfaces as sints

ppath = sints.dconfig['tilec']['plot_path']
dfact = 32

dm = sints.DR5()
shape,wcs = enmap.read_map_geometry(dm.get_map_fname('s16_01',0,False))
shape = shape[-2:]
dshape,dwcs = enmap.scale_geometry(shape, wcs, 1./dfact)

dm = sints.DR5(region_shape=shape,region_wcs=wcs)
qids = dm.adf['#qid'].to_list()
parea = maps.psizemap(dshape,dwcs)
for qid in qids:
    if qid=='p09': continue
    if qid[:3]!='s17': continue
    ivar = dm.get_ivars(qid)[:,0].sum(axis=0)
    divar = enmap.downgrade(ivar,dfact,np.sum)
    rms = maps.rms_from_ivar(divar,parea=parea)
    io.hplot(rms,f'{ppath}{qid}_frms',colorbar=True,min=4,max=100,mask=0,grid=True,ticks=20)
    io.hplot(rms,f'{ppath}{qid}_rms',colorbar=True,mask=0,grid=True,ticks=20)

