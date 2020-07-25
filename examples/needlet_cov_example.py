from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import pipeline
from soapack import interfaces as sints

dfact = 4
version = 'test_needlets'
dm = sints.DR5()
#qids = [f'boss_0{i}' for i in range(1,5)] + dm.s14_dpatches + dm.s15_dpatches + dm.s16_dpatches + dm.wide_patches
qids = dm.s14_dpatches + dm.s15_dpatches + dm.s16_dpatches + dm.wide_patches
shape,wcs = sints.get_geometry('boss_d01')
mode = 'szmode'
target_fwhm = 1.5
wmask,_ = maps.get_taper_deg(shape,wcs,3.)

def mask_fn(qid,geom=False):
    if qid in dm.wide_patches:
        if geom: return shape,wcs
        else: return wmask
    else:
        if geom: return enmap.read_map_geometry(dm.get_binary_apodized_mask_fname(qid))
        else: return dm.get_binary_apodized_mask(qid)

    
mgeos = {}
for qid in qids:
    mgeos[qid] = mask_fn(qid,geom=True)
pipeline.make_needlet_cov(version,qids,target_fwhm,mode,shape,wcs,mask_fn=mask_fn,mask_geometries=mgeos,dfact=dfact)


"""
For each needlet scale,
there are a subset of qids that get involved.
So for each scale and involved qid, we need to define a geometry for the ILC.

We start with a full sky geometry with a resolution matched for each needlet scale.
Then for each qid, we extract a sub-geometry that roughly falls in the bounding box
of that qid. We save these geometries.


"""
