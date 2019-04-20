from orphics import maps,io,cosmology,stats
from pixell import enmap,wcsutils
import numpy as np
import os,sys

def process(dm,patch,array_id,fc,mask,ncomp=1,skip_splits=False):
    """
    Return (nsplits,Ny,Nx) fourier transform
    Return (Ny,Nx) fourier transform of coadd

    This function applies no corrections for masks.
    """
    if ncomp!=1: raise NotImplementedError
    if dm.name=='act_mr3':
        season,array1,array2 = array_id.split('_')
        array = array1 + "_" + array2
    elif dm.name=='planck_hybrid':
        season,patch,array = None,None,array_id
    wins = dm.get_splits_ivar(season=season,patch=patch,arrays=[array],ncomp=None)[0,:,0,:,:]
    imaps = dm.get_splits(season=season,patch=patch,arrays=[array],ncomp=ncomp,srcfree=True)[0,:,0,:,:]
    if not(skip_splits): ksplits = fc.fft(imaps*wins*mask)
    coadd = (imaps*wins).sum(axis=0)/wins.sum(axis=0)
    coadd[~np.isfinite(coadd)] = 0
    kcoadd = enmap.enmap(fc.fft(coadd*mask),wins.wcs)
    return ksplits,kcoadd,wins


