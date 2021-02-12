from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,wcsutils
import numpy as np
import os,sys
import healpy as hp
import utils as cutils
from tilec import needlets as nd,pipeline,fg as tfg
from soapack import interfaces as sints
import h5py

# version = 'test_d4'
# lmax_file = 'data/needlet_lmaxs_test.txt'
# bound_file = 'data/needlet_bounds_test.txt'

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

fname = f"{opath}cov.h5"

mlmax = 24576
mask_alm = cs.map2alm(mask,lmax=mlmax)

tag = 'T'

def get_freq(qid):
    if qid in ['p07','p08']:
        return {'p07':353.,'p08':545}[qid]
    else:
        return {'090':98.,'150':150,'220':230}[qid[-3:]]

nfilters = bandlt.nfilters
comm,rank,my_tasks = mpi.distribute(nfilters)

for findex in my_tasks:

    fqids = bandlt.fqids[findex] # the qids in this filter
    nqids = len(fqids)

    jindex = 0
    betas = []
    crs = []
    yrs = []
    for i in range(nqids):
        for j in range(i,nqids):
            qid1 = fqids[i]
            qid2 = fqids[j]


            beta1 = bandlt.load_beta(findex,qid1,tag)
            if i!=j:
                beta2 = bandlt.load_beta(findex,qid2,tag)
            else:
                beta2 = beta1

            if jindex==0:
                bmask = cs.alm2map(mask_alm,enmap.empty(beta1.shape,beta1.wcs,dtype=np.float64))
                # io.hplot(bmask,f'{dpath}mask_findex_{findex:02d}',mask=0,grid=True,ticks=10)
                sel = np.s_[bmask>0.99]
                npix = beta1[sel].size
                cov = np.zeros((nqids,nqids,npix))
                cmap = beta1.copy()*0
                ymap = beta1.copy()*0


            fname = f"{opath}cov_findex_{findex}_{qid1}_{qid2}.fits"
            cov[i,j,:] =  enmap.read_map(fname)[sel]
            if i!=j: cov[j,i,:] =  cov[i,j,:].copy()
            jindex += 1
        betas.append(beta1[sel].copy())
        crs.append( tfg.get_mix(get_freq(qid1), 'CMB') )
        yrs.append( tfg.get_mix(get_freq(qid1), 'tSZ') )

    betas = np.asarray(betas).swapaxes(0,1)
    cov = np.moveaxis(cov,(0,1),(-2,-1))
    cinv = np.linalg.inv(cov)
    cinvk = np.einsum('...ij,...j->...i',cinv,betas)
    print(cinvk.shape)

    cnum = np.einsum('k,...k->...',np.asarray(crs)[:,0],cinvk)
    ynum = np.einsum('k,...k->...',np.asarray(yrs)[:,0],cinvk)

    cinvkc = np.einsum('...ij,j->...i',cinv,np.asarray(crs)[:,0])
    cden = np.einsum('l,...l->...',np.asarray(crs)[:,0],cinvkc)
    cinvky = np.einsum('...ij,j->...i',cinv,np.asarray(yrs)[:,0])
    yden = np.einsum('l,...l->...',np.asarray(yrs)[:,0],cinvky)

    cmap[sel] = cnum/cden
    ymap[sel] = ynum/yden

    # io.hplot(cmap,f'{dpath}cmap_findex_{findex:02d}',mask=0,grid=True,ticks=10)
    # io.hplot(ymap,f'{dpath}ymap_findex_{findex:02d}',mask=0,grid=True,ticks=10)

    fl = bandlt.filters[findex]
    flmax = bandlt.lmaxs[findex]
    fl[bandlt.ells>=flmax] = 0
    beta_alm = hp.almxfl(cs.map2alm(cmap,lmax=mlmax),fl=fl)
    dcmap = cs.alm2map(beta_alm,enmap.empty(shape,wcs,dtype=np.float32))
    # fcmap = fcmap + dcmap
    beta_alm = hp.almxfl(cs.map2alm(ymap,lmax=mlmax),fl=fl)
    dymap = cs.alm2map(beta_alm,enmap.empty(shape,wcs,dtype=np.float32))
    # fymap = fymap + dymap
    enmap.write_map(f'{dpath}dcmap_findex_{findex:02d}.fits',dcmap)
    enmap.write_map(f'{dpath}dymap_findex_{findex:02d}.fits',dymap)


        
# io.hplot(fcmap,f'{dpath}cmap',mask=0,grid=True,ticks=10)
# io.hplot(fymap,f'{dpath}ymap',mask=0,grid=True,ticks=10,color='gray')

# enmap.write_map(f'{dpath}cmap.fits',fcmap)
# enmap.write_map(f'{dpath}ymap.fits',fymap)

