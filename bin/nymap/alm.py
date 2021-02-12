from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
import healpy as hp
import utils as cutils

version = 'test90'


qids = cutils.qids
opath = f'{cutils.opath}/{version}/'

mask = enmap.read_map(f'{opath}mask.fits')


for qid in qids:
    print(qid)
    lmax = cutils.get_mlmax(qid)
    imap = enmap.read_map(f'{opath}{qid}_map.fits') * mask
    alm = cs.map2alm(imap,lmax=lmax)
    hp.write_alm(f'{opath}{qid}_alm_lmax_{lmax}.fits',alm.astype(np.complex128),overwrite=True)
