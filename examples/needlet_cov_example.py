from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from tilec import pipeline
from soapack import interfaces as sints

dfact = 4
version = 'test_needlets'
#qids =  sints.get_all_dr5_qids() #['p04','p05','p06','d6','d56_05','d56_06','s18_02']
qids = ['p04','p05','p08']#,'p06','p07','p08']
#shape,wcs = sints.get_geometry('d56_05')
shape,wcs = sints.get_geometry('d56_01')
mode = 'lensmode'
target_fwhm = 1.5
omask = sints.get_act_mr3_crosslinked_mask('deep56')
mask = enmap.extract(omask,shape,wcs)
mask_fn = lambda x: mask
mgeos = {}
for qid in qids:
    mgeos[qid] = (mask.shape,mask.wcs)
pipeline.make_needlet_cov(version,qids,target_fwhm,mode,shape,wcs,mask_fn=mask_fn,mask_geometries=mgeos,dfact=dfact)


"""
For each needlet scale,
there are a subset of qids that get involved.
So for each scale and involved qid, we need to define a geometry for the ILC.

We start with a full sky geometry with a resolution matched for each needlet scale.
Then for each qid, we extract a sub-geometry that roughly falls in the bounding box
of that qid. We save these geometries.


"""
