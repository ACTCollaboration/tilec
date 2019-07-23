from orphics import maps,io,cosmology,stats
from pixell import enmap
from actsims import inpaint as inpainting
import numpy as np
import os,sys
from soapack import interfaces as sints
from tilec import utils as tutils


def process(dm,patch,array_id,mask,skip_splits=False,splits=None,inpaint=True,fn_beam=None,cache_inpainted=False):
    """
    Return (nsplits,Ny,Nx) fourier transform
    Return (Ny,Nx) fourier transform of coadd

    This function applies no corrections for masks.
    """
    qid = array_id
    if dm.name=='act_mr3':
        season,array1,array2 = sints.arrays(qid,'season'),sints.arrays(qid,'array'),sints.arrays(qid,'freq')
        array = '_'.join([array1,array2])
        pixwin = True
    elif dm.name=='planck_hybrid':
        season,patch,array = None,None,sints.arrays(qid,'freq')
        pixwin = False
    wins = dm.get_splits_ivar(season=season,patch=patch,arrays=[array],ncomp=None)[0,:,0,:,:]
    if splits is None: 
        splits = dm.get_splits(season=season,patch=patch,arrays=[array],ncomp=3,srcfree=True)[0,:,:,:,:]
    assert splits.ndim==4
    nsplits = splits.shape[0]
    assert nsplits==2 or nsplits==4
    # Inpaint
    if inpaint and dm.name=='act_mr3':
        rsplits = []
        for i in range(nsplits): 
            result = inpainting.inpaint_map_white(splits[i],wins[i],fn_beam,union_sources_version=None,noise_pix = 20,hole_radius = 3.,plots=False)
            rsplits.append(result[0,:,:].copy())
        rsplits = enmap.enmap(np.stack(rsplits),splits.wcs)
    else:
        rsplits = splits[:,0,:,:]
    kdiffs,kcoadd = process_splits(rsplits,wins,mask,skip_splits=skip_splits,do_fft_splits=False,pixwin=pixwin)
    return kdiffs,kcoadd,wins

def process_splits(splits,wins,mask,skip_splits=False,do_fft_splits=False,pixwin=False):
    assert wins.ndim>2
    with np.errstate(divide='ignore', invalid='ignore'):
        coadd = (splits*wins).sum(axis=0)/wins.sum(axis=0)
    coadd[~np.isfinite(coadd)] = 0
    Ny,Nx = splits.shape[-2:]
    assert coadd.shape == (Ny,Nx)
    kcoadd = enmap.enmap(enmap.fft(coadd*mask,normalize='phys'),wins.wcs)
    if pixwin: pwin = tutils.get_pixwin(coadd.shape[-2:])
    else: pwin = 1
    kcoadd = kcoadd / pwin
    if not(skip_splits):
        data = (splits-coadd)*mask # !!!! wins removed, was (splits-coadd)*wins*mask earlier
        kdiffs = enmap.fft(data,normalize='phys')
        kdiffs = kdiffs / pwin
    else:
        ksplits = None
    if do_fft_splits:
        ksplits = enmap.fft(splits*wins*mask,normalize='phys')
        ksplits = ksplits / pwin
        return kdiffs,kcoadd,ksplits
    else:
        return kdiffs,kcoadd


