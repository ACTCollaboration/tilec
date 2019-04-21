from orphics import maps,io,cosmology,stats
from pixell import enmap,wcsutils
import numpy as np
import os,sys

def process(dm,patch,array_id,mask,ncomp=1,skip_splits=False):
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
    splits = dm.get_splits(season=season,patch=patch,arrays=[array],ncomp=ncomp,srcfree=True)[0,:,0,:,:]
    coadd = (splits*wins).sum(axis=0)/wins.sum(axis=0)
    coadd[~np.isfinite(coadd)] = 0
    Ny,Nx = splits.shape[-2:]
    assert splits.shape == (dm.get_nsplits(season,patch,array),Ny,Nx)
    assert coadd.shape == (Ny,Nx)
    kcoadd = enmap.enmap(enmap.fft(coadd*mask,normalize='phys'),wins.wcs)
    if not(skip_splits): 
        data = (splits-coadd)*wins*mask
        ksplits = enmap.fft(data,normalize='phys')
    else: ksplits = None
    return ksplits,kcoadd,wins


