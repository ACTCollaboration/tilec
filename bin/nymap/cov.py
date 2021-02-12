from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,wcsutils
import numpy as np
import os,sys
import healpy as hp
import utils as cutils
from tilec import needlets as nd,pipeline
from soapack import interfaces as sints
import h5py

version = 'test90'
lmax_file = 'data/needlet_lmaxs_szmode.txt'
bound_file = 'data/needlet_bounds_szmode.txt'
fwhm = 1.7
qids = cutils.qids
opath = f'{cutils.opath}/{version}/'
dpath = opath

dm = sints.DR5()

mask = enmap.read_map(f'{opath}mask.fits')
shape,wcs = mask.shape,mask.wcs
mgeos = {}
for qid in qids:
    mgeos[qid] = (shape,wcs)
bandlt = nd.BandLimNeedlet(lmax_file,bound_file,qids,dm,opath,mgeos)

"""
Take products of maps
Smooth them
That will be our covmat
"""

nfilters = bandlt.nfilters
comm,rank,my_tasks = mpi.distribute(nfilters)

for findex in my_tasks:
    fqids = bandlt.fqids[findex] # the qids in this filter
    nqids = len(fqids)
    for i in range(nqids):
        for j in range(i,nqids):
            qid1 = fqids[i]
            qid2 = fqids[j]

            beta1 = bandlt.load_beta(findex,qid1,'T')
            if i!=j:
                beta2 = bandlt.load_beta(findex,qid2,'T')
                if not(np.all(beta1.shape==beta2.shape)):
                    # we will interpret this as one map being inside the other
                    # This sets a requirement on the map geometries.
                    # FIXME: this needs to be in an assert
                    # which one is the larger one?
                    # we will run extract on it
                    if beta1.size>beta2.size:
                        beta1 = enmap.extract(beta1,beta2.shape,beta2.wcs)
                    elif beta2.size>beta1.size:
                        beta2 = enmap.extract(beta2,beta1.shape,beta1.wcs)
                    else:
                        raise ValueError
                assert np.all(beta1.shape==beta2.shape)
                assert wcsutils.equal(beta1.wcs,beta2.wcs)
                oshape = beta2.shape
                owcs = beta2.wcs
            else:
                beta2 = beta1
                oshape = beta2.shape
                owcs = beta2.wcs

            lmin = bandlt.lmins[findex]
            lmax = bandlt.lmaxs[findex]
            sm_fwhm = pipeline.get_smoothing_fwhm_arcmin(lmin,lmax)
            if sm_fwhm < 900:
                sm_filt = maps.gauss_beam(bandlt.ells,sm_fwhm)
                sht_filter_map = lambda x,fl,mlmax: cs.alm2map(hp.almxfl(cs.map2alm(x,lmax=mlmax),fl=fl),enmap.empty(oshape,owcs))
                cov = sht_filter_map(beta1 * beta2,fl=sm_filt,mlmax=max(bandlt.bounds[qid1].mlmax,bandlt.bounds[qid2].mlmax))
            else:
                print("Saving compressed form.")
                cov = beta1*0 + np.mean(beta1*beta2) / np.mean(mask**2.) # save in a compressed form instead

            # enmap.write_map_geometry(f'{dpath}cov_geometry_findex_{findex}_{qid1}_{qid2}.fits',oshape,owcs)
            fname = f"{opath}cov_findex_{findex}_{qid1}_{qid2}.fits"
            enmap.write_map(fname,cov)
            # f.require_dataset(f'findex_{findex}_{qid1}_{qid2}',data=cov,shape=cov.shape,dtype=cov.dtype,exact=True)
            # io.hplot(cov,f'{dpath}cov_{qid1}_{qid2}_findex_{findex:02d}',mask=0,grid=True,ticks=10)
