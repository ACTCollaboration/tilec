from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
import healpy as hp
import utils as cutils
from tilec import needlets as nd
from soapack import interfaces as sints


version = 'test90'
lmax_file = 'data/needlet_lmaxs_szmode.txt'
bound_file = 'data/needlet_bounds_szmode.txt'
fwhm = 1.7

qids = cutils.qids
opath = f'{cutils.opath}/{version}/'

dm = sints.DR5()


mask = enmap.read_map(f'{opath}mask.fits')
shape,wcs = mask.shape,mask.wcs
mgeos = {}
for qid in qids:
    mgeos[qid] = (shape,wcs)
bandlt = nd.BandLimNeedlet(lmax_file,bound_file,qids,dm,opath,mgeos)

nqids = len(qids)
comm,rank,my_tasks = mpi.distribute(nqids)

for i in my_tasks:
    qid = qids[i]
    print(rank,qid)
    lmax = cutils.get_mlmax(qid)
    alms = hp.read_alm(f'{opath}{qid}_alm_lmax_{lmax}.fits').astype(np.complex64)
    bandlt.transform(qid,'T',alm=alms,forward=True,target_fwhm_arcmin=fwhm)
